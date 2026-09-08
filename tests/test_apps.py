"""
tests/test_apps.py — the desktop-application layer.

The 1C tests deliberately never connect. There is exactly one infobase on
this machine and it is a client-server УТ 11 holding real trade data; a
test suite that reaches it would be a test suite that can damage the
user's business records when someone runs it with the wrong credentials
loaded. So what is tested here is the protocol -- which is where the
safety actually lives -- and the connection itself is left to a person who
knows what base they are pointing at.

That split is the point rather than a compromise. "Nothing is written
without confirmation" is a property of the state machine, and a state
machine can be tested without a database.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from mana import apps
from mana.apps import documents, onec


# ------------------------------------------------------------- capabilities

def test_available_names_what_is_missing_not_just_that_it_is():
    """A capability that is absent has to say what would make it present.

    Phase 22's lesson one level down: "unavailable" with no reason is a
    dead end for whoever reads it.
    """
    for name, info in apps.available().items():
        assert info["what"]
        if not info["available"]:
            assert info["missing"], name


def test_require_refuses_with_the_reason_attached():
    capability = apps.Capability("imaginary", "делать невозможное",
                                 modules=("no_such_module_at_all",))
    gaps = capability.missing()
    assert gaps == ["пакет no_such_module_at_all"]


def test_unknown_capability_is_refused_not_silently_allowed():
    with pytest.raises(apps.AppUnavailable):
        apps.require("no-such-capability")


def test_com_registered_distinguishes_present_from_absent():
    """Word is genuinely not installed here, and 1C genuinely is.

    Both directions matter: a checker that always says True would have
    passed a test that only looked for the installed one.
    """
    import sys
    if sys.platform != "win32":
        pytest.skip("COM только на Windows")
    assert apps.com_registered("V83.COMConnector") is True
    assert apps.com_registered("Word.Application.NoSuchThing") is False


# ---------------------------------------------------------------- documents

def test_docx_round_trip(tmp_path):
    target = tmp_path / "проба.docx"
    documents.write_docx(str(target), [
        {"kind": "heading", "text": "Отчёт", "level": 1},
        {"kind": "text", "text": "Абзац."},
        {"kind": "table", "rows": [["Товар", "Кол-во"], ["Гайка", "10"]]},
    ])
    back = documents.read_docx(str(target))
    assert back["headings"] == ["Отчёт"]
    assert "Абзац." in back["text"]
    # The table is returned as a table, not flattened into the text: a
    # table read as running prose loses which cell was in which column.
    assert back["tables"] == [[["Товар", "Кол-во"], ["Гайка", "10"]]]


def test_xlsx_round_trip(tmp_path):
    target = tmp_path / "проба.xlsx"
    documents.write_xlsx(str(target), [["Товар", "Цена"], ["Гайка", 12.5]],
                         sheet="Цены")
    back = documents.read_xlsx(str(target))
    assert back["sheet"] == "Цены"
    assert back["rows"] == [["Товар", "Цена"], ["Гайка", 12.5]]
    assert back["truncated"] is False


def test_writing_does_not_replace_an_existing_file_by_default(tmp_path):
    """An agent that silently overwrites a person's document has done
    something worse than failing."""
    target = tmp_path / "есть.xlsx"
    documents.write_xlsx(str(target), [["a"]])
    with pytest.raises(FileExistsError):
        documents.write_xlsx(str(target), [["b"]])
    documents.write_xlsx(str(target), [["b"]], overwrite=True)
    assert documents.read_xlsx(str(target))["rows"] == [["b"]]


def test_truncation_is_reported_rather_than_silent(tmp_path):
    """A shortened table that does not say it was shortened is a wrong
    table, and whoever totals it states a wrong total."""
    target = tmp_path / "много.xlsx"
    documents.write_xlsx(str(target), [[i] for i in range(50)])
    back = documents.read_xlsx(str(target), max_rows=10)
    assert len(back["rows"]) == 10
    assert back["truncated"] is True


def test_read_says_whether_values_are_cached_or_formulas(tmp_path):
    """openpyxl cannot compute a formula -- only Excel can. A caller that
    cannot tell a cached 2019 value from a fresh one will treat one as the
    other."""
    target = tmp_path / "ф.xlsx"
    documents.write_xlsx(str(target), [["=1+1"]])
    assert "формул" in documents.read_xlsx(str(target), formulas=True)["values_are"]
    assert "сохранённые" in documents.read_xlsx(str(target))["values_are"]


def test_missing_sheet_names_the_sheets_that_exist(tmp_path):
    target = tmp_path / "л.xlsx"
    documents.write_xlsx(str(target), [["a"]], sheet="Первый")
    with pytest.raises(KeyError) as caught:
        documents.read_xlsx(str(target), sheet="Нет")
    assert "Первый" in str(caught.value)


# ----------------------------------------------------------- 1C: the protocol

def _proposal(**overrides):
    fields = dict(request_id="test01", kind="set_attributes",
                  object_kind="Справочники", object_name="Номенклатура",
                  uuid="11111111-1111-1111-1111-111111111111",
                  description="изменить наименование",
                  changes={"Наименование": "Новое"},
                  before={"Наименование": "Старое"})
    fields.update(overrides)
    return onec.WriteProposal(**fields)


@pytest.fixture(autouse=True)
def _no_leftover_proposals():
    onec._PENDING.clear()
    yield
    onec._PENDING.clear()


def test_a_write_cannot_be_executed_without_a_proposal():
    with pytest.raises(onec.UnknownProposal):
        onec.confirm("никогда-не-существовавший")


def test_only_confirm_can_reach_the_executor():
    """The rule is structural, not a convention.

    `_execute` takes a WriteProposal, and the only place a caller can
    obtain one from the pending registry is `confirm`, which removes it.
    A second confirm of the same id therefore cannot re-run the write.
    """
    proposal = _proposal()
    onec._PENDING[proposal.request_id] = proposal
    # No connection, so the write fails at the COM boundary -- but it got
    # past the protocol, which is what this asserts.
    with pytest.raises(onec.OneCError):
        onec.confirm(proposal.request_id)
    # And it is gone: a replay finds nothing.
    with pytest.raises(onec.UnknownProposal):
        onec.confirm(proposal.request_id)


def test_a_stale_proposal_is_refused_not_executed():
    """It describes a database state read minutes ago. The longer it sits
    the less true that description is."""
    proposal = _proposal(created=time.time() - onec.PROPOSAL_TTL_SECONDS - 1)
    onec._PENDING[proposal.request_id] = proposal
    assert proposal.expired is True
    with pytest.raises(onec.ProposalExpired):
        onec.confirm(proposal.request_id)


def test_expired_proposals_drop_out_of_the_pending_list():
    onec._PENDING["fresh"] = _proposal(request_id="fresh")
    onec._PENDING["stale"] = _proposal(
        request_id="stale", created=time.time() - onec.PROPOSAL_TTL_SECONDS - 1)
    ids = [p["request_id"] for p in onec.pending()]
    assert ids == ["fresh"]


def test_cancelling_removes_the_proposal():
    onec._PENDING["x"] = _proposal(request_id="x")
    assert onec.cancel("x")["cancelled"] is True
    assert onec.pending() == []
    assert onec.cancel("x")["cancelled"] is False


def test_a_proposal_shows_before_and_after():
    """What a person confirms is a diff, not a verb.

    Without the current value in front of them, "изменить Наименование"
    is a request to trust the agent's reading of the database.
    """
    shown = _proposal().as_dict()
    assert shown["before"] == {"Наименование": "Старое"}
    assert shown["changes"] == {"Наименование": "Новое"}
    assert shown["expires_in"] > 0


def test_proposing_an_empty_change_is_refused():
    with pytest.raises(onec.OneCError):
        onec.propose_set_attributes("Справочники", "Номенклатура",
                                    "11111111-1111-1111-1111-111111111111", {})


# ------------------------------------------------------------ 1C: credentials

def test_the_password_never_appears_in_the_describable_form(monkeypatch):
    monkeypatch.setenv(onec.ENV_CONNECT, 'Srvr="сервер";Ref="база"')
    monkeypatch.setenv(onec.ENV_USER, "Иванов")
    monkeypatch.setenv(onec.ENV_PASSWORD, "секрет")
    shown = onec.describe_connection()
    assert "секрет" not in repr(shown)
    assert shown["password_set"] is True
    # And the credential form does carry it -- otherwise the split would
    # be pointless rather than protective.
    assert "секрет" in onec.connection_string()


def test_no_connection_string_is_a_refusal_with_the_variable_named(monkeypatch):
    monkeypatch.delenv(onec.ENV_CONNECT, raising=False)
    with pytest.raises(onec.OneCError) as caught:
        onec.connection_string()
    assert onec.ENV_CONNECT in str(caught.value)


def test_reads_refuse_without_a_connection():
    onec._THREAD._connection = None
    with pytest.raises(onec.OneCError):
        onec.query("ВЫБРАТЬ 1")


# ------------------------------------------------------------------- as tools

def test_every_app_tool_registers_and_reports_its_own_availability():
    from mana.apps.tools import register_app_tools
    from mana.tools import ToolRegistry
    registry = ToolRegistry()
    register_app_tools(registry)
    for name in ("read_document", "write_document", "open_in_editor",
                 "onec_query", "onec_metadata", "onec_propose_write",
                 "onec_confirm_write"):
        tool = registry.get(name)
        assert tool is not None, name
        assert isinstance(tool.is_available(), bool)


def test_the_document_tool_does_not_remember_the_last_file_type(tmp_path):
    """The bug this is written against: run() used to assign
    self.capability, and a registry keeps one instance for the life of the
    process -- so after the first spreadsheet, every later Word document
    was checked against the xlsx capability."""
    from mana.apps.tools import ReadDocumentTool
    sheet = tmp_path / "s.xlsx"
    document = tmp_path / "d.docx"
    documents.write_xlsx(str(sheet), [["a"]])
    documents.write_docx(str(document), [{"kind": "text", "text": "тут"}])

    tool = ReadDocumentTool()
    assert tool.run(path=str(sheet)).ok is True
    assert tool.run(path=str(document)).ok is True
    assert tool.capability == "docx"          # untouched by either call


def test_an_unsupported_extension_is_refused_by_the_document_tool():
    from mana.apps.tools import ReadDocumentTool
    result = ReadDocumentTool().run(path="заметки.txt")
    assert result.ok is False
    assert ".docx" in result.error


def test_confirm_write_is_a_separate_tool_from_propose():
    """One tool with a confirm=True argument is one hallucinated keyword
    away from writing to an accounting database."""
    from mana.apps.tools import APP_TOOLS
    names = {tool.name for tool in APP_TOOLS}
    assert "onec_propose_write" in names
    assert "onec_confirm_write" in names


def test_the_executor_is_reachable_only_from_confirm():
    """Read off the call graph, because this is the claim the whole design
    rests on: there is no second path that writes.

    A behavioural test cannot show this without a database to write to,
    and the one database on this machine holds real trade records, so the
    property is checked where it is actually enforced.
    """
    import ast
    import inspect

    from mana.apps import onec as module

    tree = ast.parse(inspect.getsource(module))
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_execute"):
                callers.append(node.name)
    assert callers == ["confirm"], callers


# ------------------------------------------------------- 1C: launching it
#
# Nothing in this section starts 1C. A test run that opened windows on the
# developer's screen would be a test run nobody lets finish, and the parts
# worth testing -- how the command line is built, and what is refused --
# do not need a running client.


IBASES_SAMPLE = """[UT11-ER]
Connect=Srvr="192.168.0.9";Ref="UT11-ER";
Version=8.3
[Информационная база]
Connect=File="C:\\1С\\Базы\\Тестовая пустая база";
Version=8.3
"""


@pytest.fixture
def fake_bases(tmp_path, monkeypatch):
    from mana.apps import onec_launch
    listing = tmp_path / "ibases.v8i"
    listing.write_text("\ufeff" + IBASES_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(onec_launch, "IBASES", listing)
    return onec_launch


def test_bases_are_read_from_the_list_1c_itself_shows(fake_bases):
    """Read, not reconstructed: this is the list the person recognises."""
    found = fake_bases.bases()
    assert [b["name"] for b in found] == ["UT11-ER", "Информационная база"]
    assert found[0]["kind"] == "клиент-сервер"
    assert found[0]["location"] == "192.168.0.9 / UT11-ER"
    assert found[1]["kind"] == "файловая"
    assert found[1]["location"].endswith("Тестовая пустая база")


def test_the_connection_string_keeps_the_quotes_1c_needs(fake_bases):
    """The bug this is written against, measured against a real 1C.

    subprocess builds a command line from a list and escapes the inner
    quotes of `File="C:\\path";` as \\", which 1C rejects with "Неверные
    или отсутствующие параметры соединения". Passing the argument without
    quotes works -- right up to the first path containing a space, and
    every base on this machine lives under C:\\1С\\Базы\\... So the line is
    built here, and a literal quote inside a quoted Windows argument is
    written twice.
    """
    line = fake_bases._connection(r"C:\1С\Базы\Новая база")
    assert line.startswith('"File=""')
    assert line.endswith('"";"')
    assert "Новая база" in line


def test_quoting_doubles_embedded_quotes(fake_bases):
    assert fake_bases._quoted('простой') == '"простой"'
    assert fake_bases._quoted('с "кавычкой"') == '"с ""кавычкой"""'


def test_launching_an_unknown_base_names_the_ones_that_exist(fake_bases):
    with pytest.raises(apps.AppUnavailable) as caught:
        fake_bases.launch(base="НетТакой")
    assert "UT11-ER" in str(caught.value)


def test_launching_without_a_base_falls_through_to_the_selector(
        fake_bases, monkeypatch):
    """"Запусти 1С" means the base list, not some base of our choosing."""
    called = []
    monkeypatch.setattr(fake_bases, "launch_selector",
                        lambda *a, **k: called.append(True) or {"launched": True})
    fake_bases.launch()
    assert called == [True]


def test_the_password_is_passed_to_1c_and_kept_out_of_the_result(
        fake_bases, monkeypatch):
    """Both halves matter.

    1C has no way to take a password except the command line, so it has to
    go there for scenario 2 to work at all. But the returned dict lands in
    a trace, and a trace is a file on disk, so the password must not be in
    it -- and the caller has to be told what the command line costs.
    """
    seen = {}

    class FakeProcess:
        pid = 4242

        def poll(self):
            return None                         # still running

    def fake_popen(line, *a, **k):
        seen["line"] = line
        return FakeProcess()

    monkeypatch.setattr(fake_bases.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(fake_bases, "_require", lambda kind: r"C:\1cv8.exe")
    # No window yet -- which is what 1C looks like while it asks for the
    # password it was just handed one of. Stubbed rather than read from
    # the real process list: a made-up pid can collide with a real
    # process, and a test that reads a stranger's window title fails on a
    # Tuesday for no reason anybody can reconstruct.
    monkeypatch.setattr(fake_bases, "_window_title", lambda pid: "")

    result = fake_bases.launch(base="UT11-ER", user="Иванов",
                               password="тайна123", settle=0.0)

    assert "тайна123" in seen["line"]          # 1C really gets it
    assert "тайна123" not in result["command"]  # the trace does not
    assert "тайна123" not in repr(result)
    assert result["warning"]                    # and the caller is told
    # Started, and honest about not knowing what opened.
    assert result["verified"] == "unobserved"


def test_no_password_means_no_warning_and_1c_will_ask(fake_bases, monkeypatch):
    class FakeProcess:
        pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(fake_bases.subprocess, "Popen",
                        lambda *a, **k: FakeProcess())
    monkeypatch.setattr(fake_bases, "_require", lambda kind: r"C:\1cv8.exe")
    monkeypatch.setattr(fake_bases, "_window_title",
                        lambda pid: "UT11-ER - 1С:Предприятие")
    result = fake_bases.launch(base="UT11-ER", user="Иванов", settle=0.0)
    assert result["warning"] == ""
    assert result["user"] == "Иванов"
    # The title names the base, so that much is confirmed. The mode is
    # not: the Предприятие window and the password dialog are titled the
    # same way, so the verdict stays honest about what was actually seen.
    assert result["outcome"]["observed"]["base"] == "UT11-ER"
    assert result["outcome"]["unobserved"] == ["mode"]


def test_creating_a_base_without_a_path_is_the_question(fake_bases):
    """The refusal IS the "где разместить?" the person expects.

    Picking a directory for them would put a database somewhere nobody
    chose, to be found later by accident.
    """
    with pytest.raises(apps.AppUnavailable) as caught:
        fake_bases.create_base()
    assert "где" in str(caught.value).lower()


def test_creating_into_a_non_empty_directory_is_refused(fake_bases, tmp_path):
    occupied = tmp_path / "занято"
    occupied.mkdir()
    (occupied / "чужой.txt").write_text("данные", encoding="utf-8")
    with pytest.raises(apps.AppUnavailable):
        fake_bases.create_base(path=str(occupied))


def test_versions_sort_numerically_not_alphabetically(fake_bases, tmp_path):
    """"8.3.9" sorts after "8.3.27" as text, and opening a new base with an
    old client fails with a message that explains nothing."""
    for name in ("8.3.9.1", "8.3.27.2214", "8.3.10.5"):
        (tmp_path / name).mkdir()
    newest = fake_bases._versions(tmp_path)[0].name
    assert newest == "8.3.27.2214"


def test_the_launch_tools_are_registered_and_available():
    from mana.apps.tools import register_app_tools
    from mana.tools import ToolRegistry
    registry = ToolRegistry()
    register_app_tools(registry)
    for name in ("onec_list_bases", "onec_launch", "onec_create_base"):
        assert registry.get(name) is not None, name


def test_launching_is_a_separate_capability_from_the_com_connection():
    """Starting the client needs the executables; reading the data needs
    pywin32 and a registered connector. Merging them would refuse
    "запусти 1С" with the wrong reason on a machine missing only COM."""
    names = {c.name for c in apps.CAPABILITIES}
    assert {"onec", "onec_client"} <= names
    client = next(c for c in apps.CAPABILITIES if c.name == "onec_client")
    assert client.modules == ()
    assert client.com_id == ""
