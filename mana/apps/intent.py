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


def _known_bases() -> List[str]:
    """The base names this machine actually has. Empty on any trouble."""
    try:
        from .onec_launch import bases
        return [str(b.get("name") or "") for b in bases() if b.get("name")]
    except Exception:
        return []


#: Cyrillic letters that are visually identical to Latin ones, plus У,
#: which is what a Russian keyboard produces where a base name has U.
#: A base called "UT11-ER" typed as "ут11-er" looks the same to a person
#: and matches nothing at all to a comparison over code points -- the same
#: trap the "1С" pattern already had to allow for.
_HOMOGLYPHS = str.maketrans(
    "АВЕКМНОРСТУХаветкмнорсух",
    "ABEKMHOPCTUXabetkmhopcux")


def _fold(text: str) -> str:
    return (text or "").lower().translate(_HOMOGLYPHS)


def _match_known_base(text: str) -> str:
    """A base from this machine's own list that the message mentions.

    Longest first: a base called "УТ" must not win over "УТ11-ER" in a
    message that names the longer one.
    """
    folded = _fold(text)
    for name in sorted(_known_bases(), key=len, reverse=True):
        if name and _fold(name) in folded:
            return name
    return ""


def _base_named(text: str) -> str:
    """Which base the message names, matched against the real list.

    Against the LIST, not against a quoting convention. The first version
    only recognised a name in quotes, so "Запусти 1С базу UT11-ER" -- how
    a person actually types it -- opened the chooser instead of the base,
    and MANA looked like it had ignored half the instruction.

    Ground truth beats a pattern here: the machine knows exactly which
    bases exist, so the question is which of those the message mentions,
    not what shape a base name has.
    """
    lowered = (text or "").lower()

    exact = _match_known_base(text)
    if exact:
        return exact

    # Nothing matched a real base. A name in quotes is still worth
    # passing on -- `onec_launch` refuses with the list of what exists,
    # which tells the person more than silently opening the chooser.
    quoted = re.search(r"[«\"']([^»\"']{2,60})[»\"']", text or "")
    if quoted:
        return quoted.group(1).strip()

    # "запусти базу <что-то>" without quotes and without a match.
    after = re.search(r"баз[уые]\s+([^\s,.;]{2,60})", lowered)
    return after.group(1).strip() if after else ""


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
        params: Dict[str, Any] = {}
        named = _base_named(text)
        if named:
            params["base"] = named
        if re.search(r"конфигуратор", head):
            params["designer"] = True
        return Intent("launch_onec", "onec_launch", params, launcher.group(0))

    # A real base name is as strong a signal as the word "1С", and it is
    # grounded rather than guessed: the name came from the machine's own
    # list. "Запусти УТ11-ER" is not ambiguous to a person and should not
    # be to this.
    named_base = _match_known_base(text)
    if named_base:
        params = {"base": named_base}
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
