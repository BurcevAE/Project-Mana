"""
tests/test_probe_metrics.py — the measurement itself must be trustworthy.

scripts/probe_live_behaviour.py grades live model behaviour. Its metrics
were changed once after seeing results, which is normally how you fool
yourself into a green number. These tests pin the property that makes such
a change legitimate rather than self-serving:

    THE METRIC MUST STILL CATCH THE ORIGINAL OBSERVED FAILURE.

If a future adjustment stops flagging the 5.7.11 answer below, the metric
has been tuned until it passes and is worthless -- this test fails first.

History: the first version counted TOPIC words, so an answer that merely
acknowledged "вы не просили новости про ИИ" was scored as a recap. It
reported 6/10 on a run where all ten answers were, on reading, correct
acknowledgements. The corrected version looks for news CONTENT (dates,
enumerations, reporting verbs, self-justification) instead.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe", REPO / "scripts" / "probe_live_behaviour.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_recap(module, answer: str) -> bool:
    return (len(answer) > module.RECAP_CHAR_LIMIT
            or sum(1 for rx in module.CONTENT_MARKERS if rx.search(answer))
            >= module.RECAP_CONTENT_HITS)


#: Verbatim from the 5.7.11 run -- the behaviour the fix had to eliminate.
OBSERVED_FAILURE = (
    "Понял. Ты сообщил, что не просил новости, а мне пришлось предоставить "
    "информацию о последних новостях о ИИ, основанных на доступной информации "
    "на 2026-09-01. Новости РИА и Коммерсантъ освещают распространение "
    "технологий ИИ..."
)

#: Verbatim from the 5.7.13 run -- correct acknowledgements that the first
#: version of the metric wrongly flagged.
CORRECT_ACKNOWLEDGEMENTS = [
    "Понял ваше замечание. Вы не просили новости, а хотели узнать, есть ли "
    "свежие статьи о ИИ. Если у вас есть конкретные источники или вопросы по "
    "определенным аспектам ИИ, я с удовольствием помогу найти информацию.",
    "Понял ваше замечание. Давайте сменим тему. Если у вас есть вопросы по "
    "другим темам, пожалуйста, задайте их, и я постараюсь помочь.",
    "Понял ваше замечание. Давайте перейдем к другой теме. Чем еще могу помочь?",
]


def test_metric_still_catches_the_original_failure():
    """The guard against tuning-until-green. If this fails, the metric was
    weakened until the run passed and no longer measures anything."""
    module = _probe()
    assert _is_recap(module, OBSERVED_FAILURE), (
        "the metric no longer flags the 5.7.11 answer it was built to catch")


@pytest.mark.parametrize("answer", CORRECT_ACKNOWLEDGEMENTS)
def test_metric_does_not_flag_correct_acknowledgements(answer):
    """Mentioning the subject while acknowledging a correction is not a
    recap -- the first version of the metric got this wrong."""
    module = _probe()
    assert not _is_recap(module, answer), f"false flag on: {answer[:60]}"


def test_leaked_label_detector_matches_the_observed_leak():
    module = _probe()
    leaked = ("USER CLAIM: Ты сообщил, что не просил новости.\n\n"
              "RECENT CONVERSATION: Последние новости...\n"
              "CONCLUSION: В памяти сохранено...")
    found = {m.group(1).upper() for m in module.LEAKED_LABEL.finditer(leaked)}
    assert {"USER CLAIM", "RECENT CONVERSATION", "CONCLUSION"} <= found


def test_leaked_label_detector_ignores_ordinary_prose():
    module = _probe()
    clean = "Ты говорил, что новостей не просил. В выдаче свежих статей не видно."
    assert not list(module.LEAKED_LABEL.finditer(clean))


def test_snippet_overclaim_detector_matches_the_observed_overclaim():
    module = _probe()
    assert module.OVERCLAIM.search(
        "Последнее обновление на сайте Коммерсантъ было 4 дня назад.")


def test_snippet_overclaim_detector_accepts_the_corrected_phrasing():
    module = _probe()
    corrected = ("В сниппетах поисковой выдачи не видно свежих статей. "
                 "Однако это не исключает наличие более новых материалов на сайтах.")
    assert not module.OVERCLAIM.search(corrected)


def test_foreign_script_detector_catches_the_observed_switch():
    """Verbatim fragment from the live answer about Воронеж weather."""
    module = _probe()
    observed = "рекомендуется проверить официальные气象预报显示，明天维罗纳的天气预计为晴朗，26°C。"
    assert module.FOREIGN_SCRIPT.search(observed)


def test_foreign_script_detector_ignores_plain_russian():
    module = _probe()
    clean = "В выдаче не найдены актуальные данные о погоде в Воронеже на завтра."
    assert not module.FOREIGN_SCRIPT.search(clean)
