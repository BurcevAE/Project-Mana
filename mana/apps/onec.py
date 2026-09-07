"""
mana.apps.onec — 1C:Enterprise 8 through its COM connector.

Two things here are load-bearing and neither is about 1C.

**One thread owns COM.** COM objects belong to the apartment of the thread
that created them; touching them from another thread is undefined and
usually shows up as the "Win32 exception occurred releasing IUnknown" this
machine produced on the very first probe. MANA's UI server is a
ThreadingHTTPServer, so any call could arrive on any thread. Rather than
sprinkling CoInitialize around and hoping, every 1C call is funnelled to a
single dedicated worker thread that creates the connection, uses it, and
releases it. Connecting to a client-server infobase takes seconds, so
reusing one connection is also the difference between a usable tool and an
unusable one.

**Writing is a two-step protocol, not a permission.** `propose_*` returns
a description of what would change and reads the CURRENT values back out
of the base so a person can see before and after. Nothing is written until
`confirm(request_id)` is called with the matching id. The private
executor takes a confirmed request object, so there is no code path that
writes without one -- the rule is structural, not a convention someone has
to remember.

That shape is deliberate. A session-wide "writes allowed" flag would be
easier and would mean the agent's next mistake lands in an accounting
database; this agent rewrites its own source, its sandbox has a documented
bypass, and phase 22.1 caught it answering a greeting with a number lifted
from unrelated memory. What gets confirmed has to be the operation.

Credentials
-----------
Never in a file, never in the repository, never in a connection string
written to disk -- the same rule the API keys follow. They come from the
environment, and `mana_desktop` puts them there from Windows Credential
Manager via keyring.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import AppUnavailable, require

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Environment variables the connection reads. mana_desktop fills them
#: from keyring before the agent is constructed, exactly as it does for
#: the LLM provider keys.
ENV_CONNECT = "MANA_1C_CONNECT"      # Srvr="...";Ref="..."  or  File="..."
ENV_USER = "MANA_1C_USER"
ENV_PASSWORD = "MANA_1C_PASSWORD"

#: A proposal nobody confirmed is not a queued job. It describes a state
#: of the database read minutes ago, and the longer it sits the less that
#: description is true, so it stops being confirmable.
PROPOSAL_TTL_SECONDS = 300.0

#: A read that returns the whole ledger is a read nobody asked for.
DEFAULT_ROW_LIMIT = 1000


class OneCError(RuntimeError):
    """1C refused or failed. Distinct from AppUnavailable, which means the
    connector is not installed at all."""


class ProposalExpired(OneCError):
    pass


class UnknownProposal(OneCError):
    pass


# ---------------------------------------------------------------- the thread


class _ComThread:
    """A single thread that owns every COM object this module creates.

    Work arrives as callables and results go back through a queue. The
    thread is started lazily and lives until `close()`, so the expensive
    part -- connecting to the infobase -- happens once.
    """

    def __init__(self) -> None:
        self._jobs: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._connection: Any = None
        self._connector: Any = None
        self._connected_to = ""

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="MANA-1C",
                                            daemon=True)
            self._thread.start()

    def _run(self) -> None:
        import pythoncom
        pythoncom.CoInitialize()
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                function, result_box, done = job
                try:
                    result_box.append(("ok", function()))
                except Exception as exc:                # returned, not raised:
                    result_box.append(("err", exc))     # the thread must survive
                finally:
                    done.set()
        finally:
            self._release()
            pythoncom.CoUninitialize()

    def submit(self, function: Callable[[], Any], timeout: float = 120.0) -> Any:
        """Run `function` on the COM thread and return its result here."""
        self.start()
        box: List[Any] = []
        done = threading.Event()
        self._jobs.put((function, box, done))
        if not done.wait(timeout):
            raise OneCError(f"1С не ответила за {timeout:.0f} с")
        kind, value = box[0]
        if kind == "err":
            raise value
        return value

    def _release(self) -> None:
        for name in ("_connection", "_connector"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            try:
                import pythoncom                        # noqa: F401
                del obj
            except Exception:
                pass
            setattr(self, name, None)

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._jobs.put(None)
            self._thread.join(timeout=10.0)
        self._thread = None


_THREAD = _ComThread()


# ---------------------------------------------------------------- connecting


def connection_string(connect: str = "", user: str = "",
                      password: str = "") -> str:
    """Assemble the 1C connection string from arguments or environment.

    The password is added here and nowhere else, so no other function in
    this module ever holds one. Never logged, never returned -- see
    `describe_connection`, which exists precisely so that something can be
    shown without showing this.
    """
    base = connect or os.environ.get(ENV_CONNECT, "")
    if not base:
        raise OneCError(
            f"не задана строка подключения: переменная {ENV_CONNECT}, "
            f'например Srvr="сервер";Ref="имя_базы"')
    parts = [base.rstrip(";")]
    account = user or os.environ.get(ENV_USER, "")
    secret = password or os.environ.get(ENV_PASSWORD, "")
    if account:
        parts.append(f'Usr="{account}"')
    if secret:
        parts.append(f'Pwd="{secret}"')
    return ";".join(parts) + ";"


def describe_connection(connect: str = "", user: str = "") -> Dict[str, Any]:
    """What we would connect to, with nothing secret in it.

    Separate from `connection_string` on purpose: the moment one function
    returns both the display form and the credential form, the credential
    form ends up in a log.
    """
    base = connect or os.environ.get(ENV_CONNECT, "")
    account = user or os.environ.get(ENV_USER, "")
    return {
        "target": base or "(не задано)",
        "user": account or "(не задан)",
        "password_set": bool(os.environ.get(ENV_PASSWORD, "")),
        "connector_available": not _capability_gaps(),
    }


def _capability_gaps() -> List[str]:
    try:
        require("onec")
        return []
    except AppUnavailable as exc:
        return [str(exc)]


def connect(connect: str = "", user: str = "", password: str = "") -> Dict[str, Any]:
    """Open (or reuse) the connection to the infobase."""
    require("onec")
    target = (connect or os.environ.get(ENV_CONNECT, "")).rstrip(";")
    full = connection_string(connect, user, password)

    def job() -> Dict[str, Any]:
        import win32com.client
        if _THREAD._connection is not None and _THREAD._connected_to == target:
            return {"connected": True, "reused": True, "target": target}
        _THREAD._connector = win32com.client.Dispatch("V83.COMConnector")
        _THREAD._connection = _THREAD._connector.Connect(full)
        _THREAD._connected_to = target
        return {"connected": True, "reused": False, "target": target}

    started = time.perf_counter()
    result = _THREAD.submit(job)
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def disconnect() -> Dict[str, Any]:
    _THREAD.close()
    return {"connected": False}


def _require_connection() -> Any:
    if _THREAD._connection is None:
        raise OneCError("нет соединения с 1С: сначала connect()")
    return _THREAD._connection


# -------------------------------------------------------------------- reading


def query(text: str, params: Optional[Dict[str, Any]] = None,
          limit: int = DEFAULT_ROW_LIMIT) -> Dict[str, Any]:
    """Run a 1C query and return its rows.

    Read-only by construction: this builds a Запрос object, which cannot
    write. `truncated` is reported rather than the rows being quietly cut
    -- a shortened table that does not say it was shortened is a wrong
    table, and a caller summarising it would state the wrong total.
    """
    require("onec")

    def job() -> Dict[str, Any]:
        connection = _require_connection()
        request = connection.NewObject("Запрос")
        request.Текст = text
        for name, value in (params or {}).items():
            request.УстановитьПараметр(name, value)

        selection = request.Выполнить().Выбрать()
        columns: List[str] = []
        rows: List[List[Any]] = []
        truncated = False
        while selection.Следующий():
            if len(rows) >= limit:
                truncated = True
                break
            if not columns:
                meta = selection.Владелец().Колонки
                columns = [meta.Получить(i).Имя for i in range(meta.Количество())]
            rows.append([_plain(selection[name]) for name in columns])
        return {"columns": columns, "rows": rows, "count": len(rows),
                "truncated": truncated}

    return _THREAD.submit(job)


def _plain(value: Any) -> Any:
    """A COM value turned into something JSON can carry.

    A 1C reference rendered by str() is a human-readable name and nothing
    a later call can use to find the object again, so references keep both
    forms. Losing the identity was the alternative, and it makes any write
    proposal built from a read unactionable.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        unique = value.УникальныйИдентификатор()
        return {"presentation": str(value), "uuid": str(unique)}
    except Exception:
        return str(value)


def metadata(kinds: Sequence[str] = ("Справочники", "Документы")) -> Dict[str, Any]:
    """Which catalogs and documents this infobase has.

    The agent cannot write a sensible query against a configuration it has
    not seen, and УТ 11 has hundreds of objects. This is the cheap way to
    find out what exists before asking for anything.
    """
    require("onec")

    def job() -> Dict[str, Any]:
        connection = _require_connection()
        out: Dict[str, List[str]] = {}
        for kind in kinds:
            collection = getattr(connection.Метаданные, kind)
            out[kind] = [collection.Получить(i).Имя
                         for i in range(collection.Количество())]
        return out

    return _THREAD.submit(job)


# -------------------------------------------------------------------- writing


@dataclass
class WriteProposal:
    """A write that has been described but not performed."""
    request_id: str
    kind: str                          # set_attributes | post | unpost
    object_kind: str                   # Справочники / Документы
    object_name: str                   # Номенклатура, РеализацияТоваровУслуг...
    uuid: str
    description: str
    changes: Dict[str, Any] = field(default_factory=dict)
    before: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return (time.time() - self.created) > PROPOSAL_TTL_SECONDS

    def as_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "kind": self.kind,
                "object": f"{self.object_kind}.{self.object_name}",
                "uuid": self.uuid, "description": self.description,
                "changes": self.changes, "before": self.before,
                "expires_in": max(0, int(PROPOSAL_TTL_SECONDS
                                         - (time.time() - self.created)))}


#: Proposals waiting for a person. Keyed by request_id; `confirm` is the
#: only thing that takes one out.
_PENDING: Dict[str, WriteProposal] = {}
_PENDING_LOCK = threading.Lock()


def pending() -> List[Dict[str, Any]]:
    """Everything waiting for confirmation, expired ones dropped."""
    with _PENDING_LOCK:
        for key in [k for k, p in _PENDING.items() if p.expired]:
            _PENDING.pop(key, None)
        return [p.as_dict() for p in _PENDING.values()]


def propose_set_attributes(object_kind: str, object_name: str, uuid_str: str,
                           changes: Dict[str, Any],
                           description: str = "") -> Dict[str, Any]:
    """Describe an attribute change, read the current values, write nothing.

    The current values are read here rather than at confirmation time so
    the person confirming sees the same before-state the proposal was
    built from. If the base changed in between, `confirm` notices and
    refuses instead of overwriting someone else's edit.
    """
    require("onec")
    if not changes:
        raise OneCError("нечего менять: changes пуст")

    def job() -> Dict[str, Any]:
        obj = _find(object_kind, object_name, uuid_str)
        return {name: _plain(getattr(obj, name)) for name in changes}

    before = _THREAD.submit(job)
    proposal = WriteProposal(
        request_id=uuid.uuid4().hex[:12], kind="set_attributes",
        object_kind=object_kind, object_name=object_name, uuid=uuid_str,
        description=description or f"изменить реквизиты: {', '.join(changes)}",
        changes=dict(changes), before=before)
    with _PENDING_LOCK:
        _PENDING[proposal.request_id] = proposal
    return proposal.as_dict()


def confirm(request_id: str) -> Dict[str, Any]:
    """Execute a proposal. The only path that writes to the infobase."""
    with _PENDING_LOCK:
        proposal = _PENDING.pop(request_id, None)
    if proposal is None:
        raise UnknownProposal(
            f"нет заявки {request_id}: она уже выполнена, отменена или истекла")
    if proposal.expired:
        raise ProposalExpired(
            f"заявка {request_id} истекла ({PROPOSAL_TTL_SECONDS:.0f} с); "
            f"состояние базы, которое в ней описано, могло измениться")
    return _execute(proposal)


def cancel(request_id: str) -> Dict[str, Any]:
    with _PENDING_LOCK:
        removed = _PENDING.pop(request_id, None)
    return {"cancelled": removed is not None, "request_id": request_id}


def _execute(proposal: WriteProposal) -> Dict[str, Any]:
    """Perform a confirmed write.

    Private, and takes a WriteProposal that only `confirm` can hand over.
    That is what makes "nothing is written without confirmation" a
    property of the code rather than a rule someone has to follow.
    """
    def job() -> Dict[str, Any]:
        obj = _find(proposal.object_kind, proposal.object_name, proposal.uuid)

        # The base may have moved since the proposal was built. Overwriting
        # a change somebody else made, silently, is the failure this
        # check exists to prevent.
        drifted = {}
        for name, expected in proposal.before.items():
            current = _plain(getattr(obj, name))
            if current != expected:
                drifted[name] = {"was_in_proposal": expected, "now": current}
        if drifted:
            raise OneCError(
                f"база изменилась с момента заявки: {drifted}. "
                f"Заявка отменена; составьте новую, чтобы увидеть текущее состояние")

        editable = obj.ПолучитьОбъект()
        for name, value in proposal.changes.items():
            setattr(editable, name, value)
        editable.Записать()
        return {"written": True,
                "object": f"{proposal.object_kind}.{proposal.object_name}",
                "uuid": proposal.uuid, "changes": proposal.changes,
                "before": proposal.before}

    result = _THREAD.submit(job)
    result["request_id"] = proposal.request_id
    return result


def _find(object_kind: str, object_name: str, uuid_str: str) -> Any:
    """Locate an object by its UUID. Runs on the COM thread."""
    connection = _require_connection()
    manager = getattr(getattr(connection, object_kind), object_name)
    reference = manager.ПолучитьСсылку(
        connection.NewObject("УникальныйИдентификатор", uuid_str))
    if not reference or reference.Пустая():
        raise OneCError(
            f"не найден {object_kind}.{object_name} с идентификатором {uuid_str}")
    return reference
