"""
mana.apps.intent — turning "открой Notepad++" into actually opening it.

The failure this exists for
----------------------------
Measured, not supposed:

    Открой Notepad++  ->  "Открываю Notepad++."   tools called: none
    Запусти 1С        ->  "Запускаю 1С."          tools called: none

Ten application tools were registered and none of them was reachable. The
answer pipeline calls a fixed list -- llm_generate, web_search,
verify_answer and a few more -- and nothing in it ever consults the
registry for a tool that matches a request. So the model narrated the
action and the action did not happen.

That is the worst failure this project can produce. Everything else here
is built so a claim rests on a measurement; an answer that says "открываю"
when nothing opened is a claim resting on nothing.

Why a matcher and not a model
------------------------------
"Запусти 1С" is not ambiguous. Asking a language model to decide would
add a dependency, a delay and a way to be wrong, to a request that a
dozen lines of pattern can settle exactly. It also means this works on a
machine with no model at all, which is the state a fresh installation is
in -- and the state the user's laptop was in when it answered every
question with a refusal.

Deliberately narrow
--------------------
Only an imperative at the start of the message counts. "Как открыть
Notepad++?" is a question about an application, not an instruction to
launch one, and a matcher that fires on both would start launching
programs at people who asked for advice. Recognising less and refusing
clearly beats guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

#: Imperatives that mean "do it", at the start of the message. A verb in
#: the middle ("расскажи, как запустить") is describing, not asking.
_LAUNCH = r"(?:запусти|запустить|открой|открыть|стартуй|включи)"

#: Written the way people write them, including the Latin/Cyrillic mix
#: that "1С" invites -- the С is Cyrillic in the product name and Latin
#: on most keyboards, and a user typing one and getting nothing would
#: reasonably call that broken.
_APPS = {
    "editor": r"(?:notepad\+\+|notepad|нотепад|блокнот\+\+)",
    "onec": r"(?:1\s*[сcС]\b|один\s*эс|1c8|1с8)",
}


@dataclass
class Intent:
    """A request that names an action, with what it names."""
    action: str                     # launch_editor | launch_onec | open_file
    tool: str                       # the registry tool that performs it
    params: Dict[str, Any] = field(default_factory=dict)
    said: str = ""                  # the fragment that matched

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "tool": self.tool,
                "params": dict(self.params), "matched": self.said}


def _path_in(text: str) -> str:
    """A Windows path mentioned in the message, if there is one."""
    found = re.search(r"[A-Za-z]:\\[^\"'<>|?*\n]+", text or "")
    return found.group(0).strip().rstrip(".,;") if found else ""


def match(task: str) -> Optional[Intent]:
    """The action this message asks for, or None.

    None is the common and correct answer: most messages are not
    instructions to launch anything, and this must stay out of the way of
    everything else.
    """
    text = (task or "").strip()
    if not text:
        return None
    head = text.lower()

    launcher = re.match(rf"^\s*{_LAUNCH}\b", head)
    if not launcher:
        return None

    path = _path_in(text)

    if re.search(_APPS["onec"], head):
        # A base named in quotes, if the message names one.
        quoted = re.search(r"[«\"']([^»\"']{2,60})[»\"']", text)
        params: Dict[str, Any] = {}
        if quoted:
            params["base"] = quoted.group(1).strip()
        if re.search(r"конфигуратор", head):
            params["designer"] = True
        return Intent("launch_onec", "onec_launch", params, launcher.group(0))

    if re.search(_APPS["editor"], head):
        if path:
            return Intent("open_file", "open_in_editor", {"path": path},
                          launcher.group(0))
        return Intent("launch_editor", "open_in_editor", {},
                      launcher.group(0))

    if path:
        # "открой C:\путь\файл.txt" without naming a program: the editor
        # is the only one this layer can open a file with.
        return Intent("open_file", "open_in_editor", {"path": path},
                      launcher.group(0))

    return None


def describe(intent: Intent, outcome: Dict[str, Any]) -> str:
    """What to tell the person, built from what actually happened.

    Never from the intent alone. The whole defect was an answer written
    from the request rather than from the result, so this function is
    only ever given an outcome to describe.
    """
    if not outcome.get("ok", False):
        return f"Не получилось: {outcome.get('error') or 'причина не названа'}"

    data = outcome.get("output") or {}
    if intent.action == "launch_editor":
        return f"Notepad++ запущен (pid {data.get('pid')})."
    if intent.action == "open_file":
        line = data.get("line")
        where = f", строка {line}" if line else ""
        return (f"Открыт в Notepad++: {data.get('path')}{where}. "
                f"Прочитать введённое обратно отсюда нельзя.")
    if intent.action == "launch_onec":
        if data.get("base"):
            return (f"Запущена база «{data['base']}» ({data.get('kind')}, "
                    f"{data.get('location')}) в режиме "
                    f"«{data.get('mode')}»."
                    + (f" {data['warning']}" if data.get("warning") else ""))
        offered = ", ".join(data.get("bases_offered") or []) or "список пуст"
        return (f"Открыто окно выбора базы 1С. Базу выбираете вы; "
                f"предложены: {offered}.")
    return "Сделано."


def perform(intent: Intent, registry: Any) -> Dict[str, Any]:
    """Call the tool this intent names, and report what it returned.

    Both branches report the truth: a refusal is passed through as a
    refusal, because "не найден Notepad++" is a useful answer and
    "открываю" would be a false one.
    """
    params = dict(intent.params)
    if intent.action == "launch_editor":
        params["launch_only"] = True
    result = registry.call(intent.tool, **params)
    return {"ok": bool(getattr(result, "ok", False)),
            "output": getattr(result, "output", None),
            "error": str(getattr(result, "error", "") or "")}
