"""
mana.apps.tools — the desktop-application capabilities, as MANA tools.

Every tool here is registered unconditionally and reports its own
availability through `is_available()`. That is the opposite of the obvious
design, which is to register only the tools whose application is
installed, and it is deliberate: a tool that is simply absent teaches the
agent nothing, while a tool that is present and answers "Word is not
installed on this machine" is a fact it can act on and tell the user
about. Phase 22 is the same lesson one level down -- a missing package
that says nothing produces an application that quietly cannot do
something.

The write tools follow the two-step protocol from `mana.apps.onec`:
`onec_propose_write` describes and reads back, `onec_confirm_write`
executes. They are registered as two separate tools rather than one tool
with a `confirm=True` argument, because a single tool with a boolean is
one hallucinated keyword away from writing to an accounting database.
"""
from __future__ import annotations

from typing import Any

from ..tools import BaseTool, ToolResult

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"


class _AppTool(BaseTool):
    """Shared plumbing: refuse clearly when the application is not here."""
    capability = ""
    requires_network = False
    requires_exec = True          # it leaves the process; that is the risk
    requires_llm = False

    @staticmethod
    def _gaps(name: str) -> list:
        from . import CAPABILITIES
        for capability in CAPABILITIES:
            if capability.name == name:
                return capability.missing()
        return [f"неизвестная возможность {name}"]

    def is_available(self, name: str = "") -> bool:
        return not self._gaps(name or self.capability)

    def _refusal(self, name: str = "") -> ToolResult:
        """Refuse by capability NAME, never by state stored on self.

        The first version assigned self.capability inside run() to switch
        between docx and xlsx. A registry keeps one tool instance for the
        life of the process, so after the first spreadsheet the instance
        checked "xlsx" for every later Word document too.
        """
        name = name or self.capability
        gaps = self._gaps(name)
        from . import CAPABILITIES
        what = next((c.what for c in CAPABILITIES if c.name == name), name)
        return ToolResult(ok=False, error=(
            f"{what} — недоступно: не хватает {', '.join(gaps)}"))


# ------------------------------------------------------------------ documents


class ReadDocumentTool(_AppTool):
    name = "read_document"
    description = ("Прочитать .docx или .xlsx: текст, заголовки, таблицы, "
                   "строки листа. Word и Excel для этого не нужны.")
    cost_hint = 0.5
    capability = "docx"

    def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", ""))
        if not path:
            return ToolResult(ok=False, error="не указан path")
        lowered = path.lower()
        from . import documents
        if lowered.endswith((".xlsx", ".xlsm")):
            if not self.is_available("xlsx"):
                return self._refusal("xlsx")
            data = documents.read_xlsx(
                path, sheet=str(kwargs.get("sheet", "")),
                max_rows=int(kwargs.get("max_rows", documents.DEFAULT_MAX_ROWS)),
                formulas=bool(kwargs.get("formulas", False)))
        elif lowered.endswith(".docx"):
            if not self.is_available("docx"):
                return self._refusal("docx")
            data = documents.read_docx(path)
        else:
            return ToolResult(ok=False, error=(
                "этот инструмент читает только .docx и .xlsx; "
                "для обычного текста есть чтение файлов"))
        return ToolResult(ok=True, output=data, meta={"path": data["path"]})


class WriteDocumentTool(_AppTool):
    name = "write_document"
    description = ("Создать .docx (список блоков: heading/text/table) или "
                   ".xlsx (список строк). Существующий файл не заменяется "
                   "без overwrite=True.")
    cost_hint = 1.0
    capability = "docx"

    def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", ""))
        if not path:
            return ToolResult(ok=False, error="не указан path")
        from . import documents
        overwrite = bool(kwargs.get("overwrite", False))
        if path.lower().endswith((".xlsx", ".xlsm")):
            if not self.is_available("xlsx"):
                return self._refusal("xlsx")
            data = documents.write_xlsx(path, kwargs.get("rows") or [],
                                        sheet=str(kwargs.get("sheet", "")),
                                        overwrite=overwrite)
        elif path.lower().endswith(".docx"):
            if not self.is_available("docx"):
                return self._refusal("docx")
            data = documents.write_docx(path, kwargs.get("blocks") or [],
                                        overwrite=overwrite)
        else:
            return ToolResult(ok=False, error="пишет только .docx и .xlsx")
        return ToolResult(ok=True, output=data, meta={"path": data["path"]})


# --------------------------------------------------------------------- editor


class OpenInEditorTool(_AppTool):
    name = "open_in_editor"
    description = ("Запустить Notepad++, при желании сразу с файлом и на "
                   "нужной строке. Показывает файл человеку; прочитать "
                   "введённое обратно нельзя.")
    cost_hint = 0.3
    capability = "editor"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import editor
        data = editor.open_file(str(kwargs.get("path", "")),
                                line=int(kwargs.get("line", 0)),
                                column=int(kwargs.get("column", 0)),
                                new_instance=bool(kwargs.get("new_instance", False)),
                                launch_only=bool(kwargs.get("launch_only", False)))
        return ToolResult(ok=True, output=data)


# ------------------------------------------------------------------------ 1C


class OneCQueryTool(_AppTool):
    name = "onec_query"
    description = ("Запрос на языке запросов 1С к базе 8.3. Только чтение: "
                   "объект Запрос писать не может.")
    requires_network = True       # client-server infobase lives on a server
    cost_hint = 2.0
    capability = "onec"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec
        text = str(kwargs.get("text", ""))
        if not text.strip():
            return ToolResult(ok=False, error="пустой запрос")
        onec.connect()
        data = onec.query(text, params=kwargs.get("params") or {},
                          limit=int(kwargs.get("limit", onec.DEFAULT_ROW_LIMIT)))
        return ToolResult(ok=True, output=data,
                          meta={"truncated": data["truncated"]})


class OneCMetadataTool(_AppTool):
    name = "onec_metadata"
    description = ("Какие справочники и документы есть в базе 1С. Нужно, "
                   "чтобы вообще знать, о чём можно спрашивать.")
    requires_network = True
    cost_hint = 1.5
    capability = "onec"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec
        onec.connect()
        kinds = kwargs.get("kinds") or ("Справочники", "Документы")
        return ToolResult(ok=True, output=onec.metadata(tuple(kinds)))


class OneCProposeWriteTool(_AppTool):
    name = "onec_propose_write"
    description = ("ОПИСАТЬ изменение реквизитов объекта 1С и прочитать "
                   "текущие значения. НИЧЕГО НЕ ПИШЕТ. Возвращает "
                   "request_id, который человек должен подтвердить.")
    requires_network = True
    cost_hint = 2.0
    capability = "onec"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec
        onec.connect()
        data = onec.propose_set_attributes(
            object_kind=str(kwargs.get("object_kind", "Справочники")),
            object_name=str(kwargs.get("object_name", "")),
            uuid_str=str(kwargs.get("uuid", "")),
            changes=kwargs.get("changes") or {},
            description=str(kwargs.get("description", "")))
        return ToolResult(ok=True, output=data,
                          meta={"request_id": data["request_id"],
                                "needs_confirmation": True})


class OneCConfirmWriteTool(_AppTool):
    name = "onec_confirm_write"
    description = ("ВЫПОЛНИТЬ ранее описанную запись в 1С по её request_id. "
                   "Вызывать только после подтверждения человеком.")
    requires_network = True
    cost_hint = 3.0
    capability = "onec"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec
        request_id = str(kwargs.get("request_id", ""))
        if not request_id:
            return ToolResult(ok=False, error="не указан request_id")
        return ToolResult(ok=True, output=onec.confirm(request_id),
                          meta={"request_id": request_id})


class OneCListBasesTool(_AppTool):
    name = "onec_list_bases"
    description = ("Какие базы 1С есть в списке на этой машине: имя, "
                   "файловая или клиент-серверная, где лежит.")
    requires_exec = False
    cost_hint = 0.2
    capability = "onec_client"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec_launch
        found = onec_launch.bases()
        return ToolResult(ok=True, output={"bases": found,
                                           "count": len(found)})


class OneCLaunchTool(_AppTool):
    name = "onec_launch"
    description = ("Запустить 1С. Без имени базы открывает окно выбора и "
                   "базу выбирает человек. С именем — открывает её; "
                   "designer=true открывает конфигуратор.")
    cost_hint = 1.0
    capability = "onec_client"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec_launch
        data = onec_launch.launch(
            base=str(kwargs.get("base", "")),
            user=str(kwargs.get("user", "")),
            password=str(kwargs.get("password", "")),
            designer=bool(kwargs.get("designer", False)),
            thin=bool(kwargs.get("thin", False)))
        # The warning rides in meta as well as output: a caller that only
        # reads meta still learns the password was exposed.
        return ToolResult(ok=True, output=data,
                          meta={"warning": data.get("warning", "")})


class OneCCreateBaseTool(_AppTool):
    name = "onec_create_base"
    description = ("Создать новую файловую базу 1С по указанному пути и "
                   "открыть конфигуратор. БЕЗ пути отказывает — спросите "
                   "у человека, где размещать.")
    cost_hint = 3.0
    capability = "onec_client"

    def run(self, **kwargs: Any) -> ToolResult:
        if not self.is_available():
            return self._refusal()
        from . import onec_launch
        data = onec_launch.create_base(
            path=str(kwargs.get("path", "")),
            name=str(kwargs.get("name", "")),
            open_designer=bool(kwargs.get("open_designer", True)),
            add_to_list=bool(kwargs.get("add_to_list", True)))
        return ToolResult(ok=True, output=data)


#: Registered by build_default_registry. Order is presentation only.
APP_TOOLS = (ReadDocumentTool, WriteDocumentTool, OpenInEditorTool,
             OneCListBasesTool, OneCLaunchTool, OneCCreateBaseTool,
             OneCQueryTool, OneCMetadataTool, OneCProposeWriteTool,
             OneCConfirmWriteTool)


def register_app_tools(registry: Any) -> None:
    """Add every desktop-application tool to a registry."""
    for tool_class in APP_TOOLS:
        registry.register(tool_class())
