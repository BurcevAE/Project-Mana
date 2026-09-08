"""
mana.cli — argparse entry point wiring Config -> ManaAgent -> (batch/interactive/voice) run modes.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pickle
import random
import re
import statistics
import sys
import threading
import time
import subprocess
import tempfile
import shutil
import platform
import ast
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .version import PRODUCT_VERSION, format_version_report, version_report
from .agent import ManaAgent
from .voice import VoiceInterface
from .optional_deps import HAS_REQUESTS, HAS_WEB, HAS_SOUNDDEVICE, sd
from . import events, paths

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "2.0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.max_pipeline_evaluations_per_cycle = max(8, args.population * max(1, args.generations))
    cfg.strategy_generations = max(1, args.generations)
    cfg.strategy_population = max(2, args.population)
    cfg.enable_web = not args.no_web and HAS_WEB
    cfg.enable_llm = not args.no_llm and HAS_REQUESTS
    cfg.voice_enabled = args.voice
    cfg.verbose_logging = args.verbose
    cfg.evolution_workers = max(1, min(4, args.evolution_workers))
    cfg.voice_whisper_model = args.voice_model
    cfg.voice_seconds = args.voice_seconds
    cfg.voice_language = args.voice_language
    cfg.voice_tts_backend = args.voice_tts_backend
    cfg.voice_silero_speaker = args.voice_speaker
    cfg.voice_output_device = args.voice_output_device
    cfg.memory_session_id = getattr(args, "session_id", "default") or "default"
    cfg.local_exec_enabled = bool(getattr(args, "enable_local_exec", False))
    cfg.local_exec_timeout = float(getattr(args, "exec_timeout", cfg.local_exec_timeout))
    if getattr(args, "web_max_retries", None) is not None:
        cfg.web_max_retries = max(0, int(args.web_max_retries))
    if getattr(args, "web_retry_delay", None) is not None:
        cfg.web_retry_delay_seconds = max(0.0, float(args.web_retry_delay))
    if getattr(args, "web_parallel", False):
        cfg.web_serialized_requests = False
    if getattr(args, "llm_model", None):
        cfg.ollama_model = args.llm_model
    if getattr(args, "llm_url", None):
        cfg.ollama_url = args.llm_url
    cfg.hardware_auto_adapt = not getattr(args, "no_hardware_adapt", False)
    # --- brain pool ---------------------------------------------------
    # NOTE: --no-llm keeps its original meaning (no local backend, see
    # BrainPool.usable) and additionally disables remote brains, because a
    # user typing --no-llm to run offline must get an offline run, not a
    # run that quietly reaches four cloud APIs instead of one local model.
    if getattr(args, "no_llm", False):
        cfg.brain_external_enabled = False
    if getattr(args, "no_external_brains", False):
        cfg.brain_external_enabled = False
    if getattr(args, "brain_policy", None):
        cfg.brain_policy = args.brain_policy
    if getattr(args, "brains_file", None):
        cfg.brains_file = args.brains_file
    if getattr(args, "allow_paid_brains", False):
        cfg.brain_allow_paid = True
    if getattr(args, "consensus", None):
        cfg.brain_consensus_n = max(2, int(args.consensus))
    return cfg


def format_brains(status: Dict[str, Any]) -> str:
    """Human-readable brain table for --list-brains.

    A brain that is configured but not ready is the interesting case (no
    key, cooling down, quota spent), so the reason is shown in-line rather
    than making the user diff two JSON blobs to find it.
    """
    lines = [f"policy={status['policy']}  готовы: {len(status['available'])}/{len(status['brains'])}", ""]
    header = f"{'BRAIN':<18}{'MODEL':<34}{'TIER':<8}{'READY':<7}{'CALLS':<7}{'LAT':<8}{'Q':<6}СТАТУС"
    lines.append(header)
    lines.append("-" * len(header))
    for b in status["brains"]:
        if b["ready"]:
            note = "ok"
        elif not b["enabled"]:
            # A brain can be off because it needs something the user has
            # not supplied yet (Cloudflare needs an account id as well as
            # a token). "Выключен" alone is a dead end -- say what is
            # missing, since that is the only reason anyone reads this
            # column.
            note = b.get("setup_hint") or "выключен"
        elif not b["key_present"]:
            note = f"нет ключа ({b['api_key_env']})"
            if b.get("setup_hint"):
                note += f" — {b['setup_hint']}"
        elif b["cooldown_for"] > 0:
            note = f"кулдаун {b['cooldown_for']:.0f}с: {b['last_error'][:40]}"
        elif b["rpd"] and b["day_count"] >= b["rpd"]:
            note = f"исчерпан дневной лимит ({b['rpd']})"
        elif not b["usable"]:
            note = "недоступен (сеть/режим)"
        else:
            note = "лимит запросов в минуту"
        lines.append(f"{b['brain_id']:<18}{b['model'][:33]:<34}{b['tier']:<8}"
                     f"{('да' if b['ready'] else 'нет'):<7}{b['calls']:<7}"
                     f"{b['ewma_latency']:<8.2f}{b['ewma_quality']:<6.2f}{note}")
    return "\n".join(lines)


#: How a call is stamped when the record is read back. "!" is the call
#: itself failing, which was the only state this printer had; the other
#: two come from the tool having looked at the machine afterwards, and
#: telling them apart is the point of mana.outcome.
_MARKS = {"contradicted": "✗", "unobserved": "?"}


def _mark(call) -> str:
    if not call.ok:
        return "!"
    return _MARKS.get(call.verified, "")


def _show_journal(limit: int) -> int:
    """Print what the last turns actually did.

    The tool list per turn is the column worth reading. An answer that
    says "открываю" beside an empty tool list is the failure this record
    was created for -- and this only shows it, deliberately: judging is a
    separate step with its own decisions, and a viewer that graded turns
    would be asserting a verdict nobody has yet defined.
    """
    from .journal import Journal

    journal = Journal()
    stats = journal.stats()
    if not stats["exists"]:
        print(f"Журнала ещё нет: {stats['path']}")
        print("Он пишется при обычной работе — задайте пару вопросов и вернитесь.")
        return 0

    print(f"{stats['path']}")
    print(f"эпизодов: {stats['episodes']}   сессий: {stats['sessions']}   "
          f"без единого вызова инструмента: {stats['with_no_calls']}")
    print(f"по маршрутам: {stats['by_route']}")
    if stats["by_tool"]:
        top = list(stats["by_tool"].items())[:8]
        print("инструменты: " + ", ".join(f"{n}×{c}" for n, c in top))
    print()

    for ep in journal.episodes(limit=limit):
        when = time.strftime("%d.%m %H:%M", time.localtime(ep.started))
        tools = ", ".join(f"{c.tool}{_mark(c)}" for c in ep.calls)
        print(f"[{when}] {ep.route:<13} {ep.latency:5.1f}s")
        print(f"  запрос:       {ep.request[:150]}")
        print(f"  ответ:        {ep.answer[:150]}")
        print(f"  инструменты:  {tools or '(ни одного)'}")
        print()
    return 0


def _show_findings(limit: int) -> int:
    """Report the invariant violations in the recorded episodes.

    Shadow: this reads a record already written and changes nothing. The
    two kinds are printed apart on purpose -- `mechanical` is decided by
    comparing recorded fields, `pattern` is a word list making a guess,
    and presenting a guess as a fact is the failure this project keeps
    having to fix.
    """
    from .journal import Journal
    from .cognition.invariants import scan, summarise, MECHANICAL

    journal = Journal()
    episodes = journal.episodes(limit=limit)
    if not episodes:
        print(f"Журнала ещё нет или он пуст: {journal.path}")
        return 0

    violations = scan(episodes)
    summary = summarise(episodes, violations)
    print(f"Просмотрено ходов: {summary['episodes']}   "
          f"с нарушением: {summary['episodes_with_a_violation']}   "
          f"всего нарушений: {summary['violations']}")
    print(f"по инвариантам: {summary['by_invariant'] or '(ничего)'}")
    if summary["episodes"] < 30:
        print("Выборка мала: как оценка доли это число ничего не значит.")
    print()

    known = {ep.episode_id: ep for ep in episodes}
    for kind, title in ((MECHANICAL, "ИЗМЕРЕНО (сравнение записанных полей)"),
                        ("pattern", "ПО ШАБЛОНУ (догадка, проверьте глазами)")):
        chosen = [v for v in violations if v.kind == kind]
        if not chosen:
            continue
        print(f"── {title} ── {len(chosen)}")
        for violation in chosen:
            episode = known.get(violation.episode_id)
            when = (time.strftime("%d.%m %H:%M", time.localtime(episode.started))
                    if episode else "?")
            print(f"  [{when}] {violation.invariant}")
            if episode is not None:
                print(f"    запрос: {episode.request[:120]}")
                print(f"    ответ:  {episode.answer[:120]}")
            print(f"    почему: {violation.reason}")
            print()
    if not violations:
        print("Ни один инвариант не нарушен.")
    return 0


def _show_proposals(limit: int) -> int:
    """Print the changes MANA proposes for itself, ranked by a dry run.

    Nothing here adopts anything. The ordering says what is worth
    measuring properly, not what to ship -- acceptance belongs to
    core/gates.py, on evidence.
    """
    from .journal import Journal
    from .cognition import invariants, candidates, failure_domain, lessons

    journal = Journal()
    episodes = journal.episodes(limit=limit)
    if not episodes:
        print(f"Журнала ещё нет или он пуст: {journal.path}")
        return 0

    violations = invariants.scan(episodes)
    # What the record knows, not only what this window holds: a failure
    # seen fourteen times in real work is worth addressing on a day when
    # the last twenty turns happen to be clean.
    learned = lessons.read(invariants_seen=[v.invariant for v in violations])
    if not violations and not any(l.observed for l in learned.values()):
        print(f"Просмотрено ходов: {len(episodes)}. Нарушений нет — "
              f"предлагать нечего.")
        return 0

    situations = failure_domain.situations_from(episodes)
    made = candidates.plan(violations, lessons=learned)
    rows = candidates.rank(violations, situations, lessons=learned)

    if learned:
        print("что известно об отказах:")
        for lesson in sorted(learned.values(), key=lambda l: -l.observed):
            print("  " + lesson.describe())
        print()
    if made.refusals:
        print("на что настройкой не ответить:")
        for refusal in made.refusals:
            print(f"  «{refusal['addresses']}» ({refusal['observed_failures']}): "
                  f"{refusal['why']}")
        print()

    print(f"Ходов: {len(episodes)}   нарушений: {len(violations)}   "
          f"кандидатов: {len(rows)}")
    print("Оценка всухую — верхняя граница: считается, что выполнимое "
          "действие удаётся.")
    print()

    for row in rows:
        dry = row.get("dry") or {}
        if dry.get("dry_evaluable"):
            gain = dry["candidate_pass_rate"] - dry["baseline_pass_rate"]
            head = (f"{dry['baseline_pass_rate']:.2f} -> "
                    f"{dry['candidate_pass_rate']:.2f}  "
                    f"({gain:+.2f}, сломано: "
                    f"{dry['counterexamples']['found']})")
        else:
            head = "всухую не оценивается"
        print(f"  {head}")
        print(f"    менять:  {row['changes']}")
        print(f"    целит в: {row['addresses']}  ({row['observed_failures']} наруш.)")
        print(f"    почему:  {row['rationale'][:150]}")
        if not dry.get("dry_evaluable") and dry.get("reason"):
            print(f"    оценка:  {dry['reason'][:150]}")
        print()

    print("Ни одно из этих изменений не применено. Вердикт выносят ворота "
          "по свидетельствам, а их пока недостаточно.")
    return 0


def _show_capabilities() -> int:
    """What MANA can do beyond what it shipped with, and whether it is
    proved. A capability that is installed but failed its checks is
    reported as REFUSED and must not be used: a rules engine that is
    quietly wrong is a lying oracle, and every measurement built on it is
    poisoned invisibly."""
    from .acquire import status_all

    for reported in status_all():
        verification = reported["verification"]
        print(f"{reported['name']}: {reported['what']}")
        print(f"  {reported['describe']}")
        print(f"  истина из: {reported['truth_source']}")
        for check in verification.get("checks", []):
            mark = "OK  " if check["ok"] else "СБОЙ"
            print(f"    {mark} {check['name']}: "
                  f"ждали {check['expected']}, получили {check['got']}")
        if verification["status"] == "ABSENT":
            print(f"  установить: MANA.exe --acquire {reported['name']} --yes")
        print()
    return 0


def _acquire_capability(name: str, consented: bool) -> int:
    """Install a declared provider, only when a person said so.

    `pip install` executes code from the package, so consent is the line
    between self-improving and self-compromising. It is a separate word
    on the command line rather than a prompt, so it also cannot be
    implied by a script or a hook.
    """
    from .acquire import capability, install, packages_dir, ConsentRequired

    known = capability(name)
    if known is None:
        print(f"Способность «{name}» не объявлена. Доступные: "
              f"MANA.exe --capabilities")
        return 2

    if not consented:
        provider = known.providers[0]
        print(f"Способность:  {known.name} — {known.what}")
        print(f"Поставщик:    пакет «{provider.package}» ({provider.why})")
        print(f"Проверка:     {known.truth_source}")
        print(f"Куда:         {packages_dir()}")
        print()
        print("pip install исполняет код из пакета. Ничего не установлено.")
        print(f"Если согласны:  MANA.exe --acquire {name} --yes")
        return 0

    print(f"Устанавливаю {known.providers[0].package}...")
    try:
        result = install(name, consented=True,
                         on_line=lambda line: print(f"  {line[:120]}"))
    except ConsentRequired as exc:
        print(str(exc))
        return 2
    print()
    if not result.get("ok"):
        print("Не вышло: " + str(result.get("error") or result.get("describe")))
        for check in (result.get("verification") or {}).get("checks", []):
            if not check["ok"]:
                print(f"  СБОЙ {check['name']}: ждали {check['expected']}, "
                      f"получили {check['got']}")
        return 1
    print(result["describe"])
    return 0


def _lichess_token() -> int:
    """Store the Lichess token, without it ever being a visible string.

    Typed rather than passed as an argument: a command line is readable
    by any process running as this user, which is the same reason 1C
    passwords are never put on one. `getpass` does not echo it, it does
    not reach the shell history, and it goes straight to Windows
    Credential Manager -- never to a file and never to the repository.
    """
    import getpass

    from .net import lichess

    print(f"Токен lichess.org (не отображается). Пустая строка — удалить "
          f"сохранённый.\n"
          f"Права, которые нужны: {', '.join(lichess.SCOPES)}")
    try:
        value = getpass.getpass("токен: ")
    except (EOFError, KeyboardInterrupt):
        print("\nотменено")
        return 1
    result = lichess.save_token(value)
    if not result["ok"]:
        print(result["error"])
        return 1
    if not result["stored"]:
        print("сохранённый токен удалён")
        return 0
    state = lichess.Lichess().describe()
    print(f"сохранён в диспетчере учётных данных ({lichess.TOKEN_ENV})")
    if state.get("error"):
        print(state["error"])
        return 1
    print(f"аккаунт: {state.get('user', '?')}   "
          f"бот: {'да' if state.get('bot') else 'нет'}")
    if state.get("note"):
        print(state["note"])
    return 0


def _lichess(games: int, port: int = 0, watch: bool = True) -> int:
    """Play on Lichess, with a window showing what the search was thinking.

    Without a number this reports what this machine can do and stops. The
    account upgrade is not among the things it can do: turning an account
    into a bot is irreversible and belongs to the person whose account it
    is, so it is printed as a command rather than performed.
    """
    from .cognition import chess_bot, chess_watch
    from .net import lichess

    client = lichess.Lichess()
    state = client.describe()
    print(f"токен: {'есть' if state['token'] else 'нет'} ({state['env']})")
    if "user" in state:
        print(f"аккаунт: {state['user']}   бот: "
              f"{'да' if state.get('bot') else 'нет'}")
    if state.get("error"):
        print(f"Lichess: {state['error']}")
    if state.get("note"):
        print(state["note"])
    if not state.get("bot"):
        return 1
    if games <= 0:
        path = chess_bot.games_path()
        played = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
        print(f"сыграно и записано: {played}   {path}")
        print("Играть:  MANA.exe --lichess 1")
        return 0

    httpd = None
    if watch:
        try:
            httpd = chess_watch.start(port or chess_watch.DEFAULT_PORT)
            print(f"наблюдение: {chess_watch.url(httpd)}")
        except OSError as exc:
            print(f"окно не открылось ({exc}); играю без него")

    bot = chess_bot.Bot(client=client)
    events.install_console_sink()
    print(f"Жду вызова на lichess.org/@/{state['user']} "
          f"(или бросьте вызов сами). Ctrl+C — выход.")
    try:
        seats = bot.run(games=games)
    except KeyboardInterrupt:
        bot.stop()
        seats = bot.finished
    finally:
        if httpd is not None:
            httpd.shutdown()
    for seat in seats:
        row = chess_bot.summarise(seat)
        print(f"{seat.url}  {seat.status} {seat.winner}  "
              f"ходов {row['moves']}, ошибок {row['mistakes']}, "
              f"зевков {row['blunders']}, жребием "
              f"{row['close_share']:.0%}")
    return 0


def _practice(games: int) -> int:
    """Play games against itself and add them to the corpus.

    Refuses outright when the rules engine is not verified. A corpus
    generated on a subtly wrong engine is wrong in a way nothing
    downstream can detect, so this is a refusal rather than a warning.
    """
    from .cognition import chess_arena as arena

    corpus = arena.Corpus()
    if games <= 0:
        stats = corpus.stats()
        if not stats["exists"]:
            print(f"Корпуса ещё нет: {stats['path']}")
            print("Сыграть партии:  MANA.exe --practice 50")
            return 0
        print(f"{stats['path']}")
        print(f"партий: {stats['games']}   позиций: {stats['positions']}")
        print("Независимых наблюдений здесь столько же, сколько партий: "
              "позиции одной партии делят дебют и исход.")
        print(f"по результату: {stats['by_result']}")
        print(f"по причине:    {stats['by_reason']}")
        print(f"суммарно:      {stats['plies_total']} полуходов, "
              f"{stats['seconds_total']:.0f}с игры")
        return 0

    try:
        chess = arena.oracle()
    except arena.Unverified as exc:
        print(f"Играть нельзя: {exc}")
        print("Приобрести правила:  MANA.exe --acquire chess_rules --yes")
        return 1

    player = arena.SearchPlayer(depth=2, name="material-d2")
    before = len(corpus.games())
    print(f"Играю {games} партий ({player.name} против себя)...")

    played = []
    def note(game):
        played.append(game)
        if len(played) % 10 == 0:
            print(f"  {len(played)}/{games}", flush=True)

    batch = arena.self_play(player, games, seed=before * 1000, chess=chess,
                            on_game=note)
    corpus.append(batch)
    stats = corpus.stats()
    print()
    print(f"Сыграно {len(batch)}. В корпусе: {stats['games']} партий, "
          f"{stats['positions']} позиций.")
    print(f"по результату: {stats['by_result']}")
    return 0


def _show_tried() -> int:
    """Print the findings ledger: what was tried and what came of it.

    A negative result is the valuable one. "We tried this and it did not
    work" saves more time than the positive case, which is usually
    already visible in the behaviour, and it is invisible by construction
    unless somebody writes it down.
    """
    from .cognition.findings import Ledger

    ledger = Ledger()
    stats = ledger.stats()
    if not stats["exists"]:
        print(f"Реестра находок ещё нет: {stats['path']}")
        print("Он заполняется, когда эксперимент доходит до вердикта.")
        return 0

    print(f"{stats['path']}")
    print(f"экспериментов: {stats['experiments']}   записей: {stats['records']}")
    print(f"по вердиктам:  {stats['by_verdict']}")
    print()
    for finding in ledger.latest():
        print(finding.describe())
        moved = ", ".join(f"{k}={v}" for k, v in sorted(finding.conditions.items()))
        if moved:
            print(f"  условия: {moved}")
        if finding.note:
            print(f"  замечание: {finding.note[:400]}")
        print()
    print("Находка — это основание ожидать исхода, а не запрет проверять "
          "снова: если условия сдвинулись, эксперимент имеет смысл повторить.")
    return 0


def _show_series() -> int:
    """Print each question's run of findings and where the class changed.

    It reports differences between recorded rows and explains nothing. A
    reader that said "increasing the corpus improves the result" would be
    making a causal claim from two points and putting it into the record
    with the authority of an observation.
    """
    from .cognition import series

    runs = series.all_series()
    if not runs:
        print("Реестр находок пуст — серий нет.")
        print("Он заполняется, когда эксперимент доходит до вердикта.")
        return 0

    for run in runs:
        print(run.describe())
        summary = run.summary()
        if summary["confounded_flips"]:
            print(f"  переворотов с несколькими изменёнными условиями: "
                  f"{summary['confounded_flips']} — приписать перемену "
                  f"одному условию нельзя")
        if summary["unusable"]:
            print(f"  пар, которые нельзя сравнить: {summary['unusable']}")
        print()
    print("Серия показывает, ГДЕ класс изменился и ЧТО при этом отличалось. "
          "Почему — она не говорит.")
    return 0


def _next_policy_experiment() -> int:
    """The policy setting worth measuring next, chosen rather than listed.

    The half of "what next" that can be answered: a knob is controllable
    by construction -- declared in `policy.KNOBS` with its options -- and
    how much measuring one would tell us comes out of the ledger. The
    choice goes through `experiments.select`, the same selector that picks
    a probe and a pipeline experiment.
    """
    from .journal import Journal
    from .cognition import candidates, failure_domain, invariants, lessons

    episodes = Journal().episodes(limit=200)
    violations = invariants.scan(episodes) if episodes else []
    learned = lessons.read(invariants_seen=[v.invariant for v in violations])
    if not violations and not any(l.observed for l in learned.values()):
        return 0

    situations = failure_domain.situations_from(episodes) if episodes else []
    rows = candidates.rank(violations, situations, lessons=learned)
    picked = candidates.choose(violations, situations, budget=10 ** 6,
                               lessons=learned)

    print("настройка политики — здесь выбор возможен:")
    if picked is None:
        made = candidates.plan(violations, lessons=learned)
        for refusal in made.refusals:
            print(f"  «{refusal['addresses']}»: {refusal['why']}")
        if not made.refusals:
            print("  ничего не проходит порог ценности — это тоже ответ")
        print()
        return 0

    print(f"  измерить: {picked['changes']}")
    print(f"  целит в:  {picked['addresses']} "
          f"({picked['observed_failures']} наблюдений)")
    print(f"  ценность: {picked['value']:+.3f} "
          f"(информативность {picked['information']:.2f})")
    print(f"  почему:   {picked['information_why']}")
    for row in rows[1:4]:
        print(f"    следом: {row['changes']} — ценность {row['value']:+.3f}")
    print()
    return 0


def _show_next() -> int:
    """Which condition is worth varying next, per question.

    It ranks axes and delegates the choice to `experiments.select`, the
    same selector and the same floor everything else is held to. Cost is
    the caller's fact, so nothing is priced here and every axis comes back
    unpriced -- visible rather than assumed cheap.
    """
    from .cognition import lawgiver, probes, series

    _next_policy_experiment()

    runs = series.all_series()
    if not runs:
        print("Серий в реестре нет — по условиям опытов предлагать нечего.")
        return 0

    # A standing law with an untested limit lifts the axis it claims
    # about. That is what PROPOSED is licensed to do -- point at the next
    # experiment -- and it may not change how anything answers.
    book = lawgiver.load_book()
    for run in runs:
        print(f"вопрос: {run.question}")
        print(f"наблюдений: {len(run.observations)}   домен: {run.domain or '—'}")
        if len(run.observations) < 2:
            print("  серия из одного наблюдения — оси ещё не сравнивались")
        for probe in probes.probes(run, book=book):
            print("  " + probe.describe())
        print()

    print("Цена не задана: чем обходится опыт — факт предметной области, "
          "и придумывать его здесь значило бы придумывать измерение.")
    print("Управляемые оси не объявлены, поэтому список включает и "
          "контекстные условия (версии компонентов). Это ранжирование, "
          "а не выбор: отсеять их — задача того, кто ставит опыт.")
    return 0


def _run_cycle() -> int:
    """Walk the whole loop over the record and say where it stopped.

    Every stage of this existed and each was reached by a different flag,
    so the loop was one a person had to walk. What it prints last is what
    stopped it: there is nearly always something, and a loop that could
    not name the stage it stalled at would look exactly like one that
    quietly did nothing.

    The result goes into the ledger. A cycle whose outcome is not written
    down has to be re-run to be remembered, and re-running is how a
    rejected change comes back next week as a new idea.
    """
    from .journal import Journal
    from .cognition import cycle

    journal = Journal()
    episodes = journal.episodes()
    ran = cycle.run(episodes)
    print(ran.describe())

    replay = ran.stage(cycle.REPLAY)
    if replay:
        print()
        print(f"  доля прохождения: {replay.detail['baseline_pass_rate']} → "
              f"{replay.detail['candidate_pass_rate']}")
        print(f"  починено: {replay.detail['fixed'] or '—'}")
        print(f"  сломано:  {replay.detail['broke'] or '—'}")
    if ran.finding_id:
        print()
        print(f"записано в реестр: {ran.finding_id}")
    print()
    print("Сухой прогон — верхняя граница: считается, что выполнимое "
          "действие удаётся. Живой результат может быть только хуже.")
    return 0


def _run_world(steps: int) -> int:
    """Explore the small world and score the model that comes out.

    An experiment, not a feature: nothing in the running agent consults a
    world model, and this reports how much of one can be recovered from
    acting in a world whose rules are known. The two failure kinds are
    printed apart -- what was never established and what was invented --
    because a missing condition promises too much and an invented one
    refuses work that would have worked.
    """
    from .cognition import acting, lawgiver
    from .world.explore import Explorer
    from .world.universe import SmallWorld, grade

    # The one place a law is allowed to change what is done rather than
    # what is measured next. Standing laws only -- a PROPOSED one may
    # point at the next experiment and no further.
    from .world.explore import EPISODE_STEPS

    chosen = acting.setting_for("steps", steps, domain="world_model",
                                book=lawgiver.load_book(),
                                # The conditions this run is under. A law
                                # measured at another episode length was
                                # measured somewhere else, and saying so
                                # is what keeps it from speaking here.
                                conditions={"episode_steps": EPISODE_STEPS})
    if chosen.changed:
        print("по закону: " + chosen.describe())
        steps = int(chosen.value)

    world = SmallWorld(seed=7)
    model = Explorer().explore(world, steps=steps, seed=7).model()
    print(model.describe())
    print()
    print(f"=== сверка с настоящими правилами мира ({steps} шагов) ===")
    print(grade(model).describe())
    print()
    print("Это опыт над самой идеей модели мира, а не возможность агента: "
          "живой путь ответа никакую модель мира не спрашивает.")
    return 0


def _show_laws() -> int:
    """The law book: conditional claims and the status the evidence earns.

    Statuses are derived, never set. PROPOSED means one supported
    experiment and nothing beyond it -- a summary of counts alone would
    let that read as "almost a law".
    """
    from .cognition import lawgiver, series
    from .cognition.laws import LawBook

    book = lawgiver.load_book()
    made = []
    refused = []
    for run in series.all_series():
        made.extend(lawgiver.propose(run, book))
        refused.extend((run.question, r) for r in lawgiver.assess(run).refusals)
    if made:
        lawgiver.save_book(book)

    # A series that produced nothing used to look exactly like a series
    # nobody had. Now that there is a bar to clear, the reason a claim did
    # not clear it is usually the interesting half of the output.
    if refused:
        print("на что серии пока не тянут:")
        for question, refusal in refused:
            print(f"  [{question[:44]}] {refusal.describe()}")
        print()

    if not book.all():
        print("Законов пока нет.")
        print(f"Закон рождается из СЕРИИ изолированных переворотов: одно "
              f"условие изменилось, класс изменился, метод тот же — и так "
              f"минимум {lawgiver.MIN_AGREEING_FLIPS} раза, согласно.")
        print("Смотреть серии:  MANA.exe --series")
        return 0

    for law in book.all():
        print(law.describe())
        for note in law.exceptions:
            print(f"    исключение: {note}")
        if law.history:
            for step in law.history:
                print(f"    статус {step['from']} -> {step['to']} "
                      f"на {step['trials']} испытаниях")
        print()

    reported = lawgiver.report(book)
    print(f"по статусам: {reported['by_status']}")
    print(reported["note"])
    return 0


def _forget_junk(consented: bool) -> int:
    """Remove stored items the guards would refuse to write today.

    Reports first and removes only when told to. Deleting somebody's
    memory is not something to do quietly, and a cleanup that reports a
    count without the rows is one nobody can check.
    """
    from .config import Config
    from . import ManaAgent

    cfg = Config(enable_llm=False, enable_web=False)
    cfg.ensure_dirs()
    graph = ManaAgent(cfg).graph_memory
    report = graph.forget_junk(consented=consented)

    nodes, entities = report["nodes"], report["entities"]
    if not nodes and not entities:
        print("Мусора в памяти не найдено.")
        return 0

    print(f"Записи, которые сегодняшние ограждения не пропустили бы: "
          f"{len(nodes)} узлов, {len(entities)} осиротевших сущностей")
    print()
    for row in nodes:
        print(f"  узел      {row['text']}")
    for row in entities:
        print(f"  сущность  {row['text']}")
    print()
    if report.get("error"):
        print("Не удалось удалить: " + report["error"])
        return 1
    if report["removed"]:
        print("Удалено.")
        return 0
    print("Ничего не удалено. Если согласны:  MANA.exe --forget-junk --yes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser, split out of main() so tests can construct it
    without running the agent."""
    parser = argparse.ArgumentParser(description=f"MANA {PRODUCT_VERSION}")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--self-improve", type=int, default=0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--evolution-workers", type=int, default=2, choices=range(1, 5), metavar="N")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--routing-benchmark", action="store_true", help="Проверить AUTO/LOCAL/WEB/MIXED routing и реальное выполнение web")
    parser.add_argument("--routing-holdout", action="store_true", help="Запустить изолированный routing holdout")
    parser.add_argument("--adaptive-benchmark", action="store_true", help="Проверить adaptive compute / confidence / architecture")
    parser.add_argument("--adaptive-holdout", action="store_true", help="Запустить изолированный adaptive-compute holdout")
    parser.add_argument("--adaptive-repetitions", type=int, default=None, help="Количество повторов adaptive benchmark")
    parser.add_argument("--adaptive-holdout-repetitions", type=int, default=None, help="Количество повторов adaptive holdout")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--enable-local-exec", action="store_true", help="Разрешить запуск небольших локальных Python-проверок в sandbox; без повышения прав")
    parser.add_argument("--exec-timeout", type=float, default=3.0, help="Макс. время локальной проверки, секунд")
    parser.add_argument("--web-max-retries", type=int, default=None, help="Макс. повторов Web-запроса при transport error")
    parser.add_argument("--web-retry-delay", type=float, default=None, help="Задержка между повторными Web-запросами")
    parser.add_argument("--web-parallel", action="store_true", help="Разрешить параллельные Web-запросы в evolution (по умолчанию запросы сериализуются)")
    parser.add_argument("--verify", metavar="EXPR", default=None, help="Проверить арифметическое выражение локально")
    parser.add_argument("--run-code", metavar="CODE_OR_FILE", default=None, help="Запустить Python-файл или inline-код в sandbox (требует --enable-local-exec)")
    parser.add_argument("--session-id", default="default", help="Идентификатор постоянной сессии диалога")
    parser.add_argument("--memory-search", metavar="QUERY", default=None, help="Семантический поиск по долговременной памяти")
    parser.add_argument("--memory-status", action="store_true", help="Показать состояние persistent memory")
    parser.add_argument("--learn", metavar="PATH_OR_TOPIC", default=None, help="Изучить файл, каталог или тему через Web")
    parser.add_argument("--learn-topic", metavar="TOPIC", default=None, help="Изучить тему через Web")
    parser.add_argument("--learn-domain", default="", help="Домен знания, например 1c")
    parser.add_argument("--learn-nonrecursive", action="store_true", help="Не обходить подкаталоги")
    parser.add_argument("--llm-model", default=None, metavar="NAME",
                        help="Имя модели LLM-бэкенда (для ollama — то, что показывает `ollama list`, "
                             "например 'mana' или 'qwen2.5:7b-instruct-q4_K_M'). "
                             "По умолчанию берётся Config.ollama_model.")
    parser.add_argument("--llm-url", default=None, metavar="URL",
                        help="Endpoint ollama (по умолчанию http://localhost:11434/api/generate).")
    parser.add_argument("--version", action="store_true", help="Показать версию продукта и всех подсистем")
    parser.add_argument("--knowledge-status", action="store_true", help="Показать статистику базы знаний")
    parser.add_argument("--hardware-status", action="store_true", help="Показать определённый профиль машины и что было адаптировано")
    parser.add_argument("--no-hardware-adapt", action="store_true", help="Не подстраивать Config под текущую машину")
    parser.add_argument("--list-tools", action="store_true", help="Показать зарегистрированные инструменты агента")
    # --- brain pool (mana.brains) ---
    parser.add_argument("--paths-status", action="store_true",
                        help="Где MANA ищет состояние, песочницу и собственный код "
                             "(первое, что нужно смотреть, если память 'потерялась')")
    parser.add_argument("--journal", nargs="?", const=20, type=int, metavar="N",
                        help="Что MANA реально делала на последних N ходах: "
                             "запрос, вызванные инструменты, ответ")
    parser.add_argument("--findings", nargs="?", const=200, type=int, metavar="N",
                        help="Прогнать инварианты по последним N ходам журнала "
                             "и показать найденные отказы (ничего не меняет)")
    parser.add_argument("--propose", nargs="?", const=200, type=int, metavar="N",
                        help="Какие изменения MANA предлагает себе по последним "
                             "N ходам журнала (ничего не применяет)")
    parser.add_argument("--laws", action="store_true",
                        help="Законы: условные утверждения и статус, который "
                             "им дало накопленное свидетельство")
    parser.add_argument("--next", action="store_true", dest="next_probe",
                        help="Какое условие стоит поварьировать дальше и почему")
    parser.add_argument("--cycle", action="store_true",
                        help="Полный круг: опыт → ошибка → исследование → "
                             "изменение → повторная ситуация → результат, "
                             "и чего не хватает, чтобы он замкнулся")
    parser.add_argument("--world", nargs="?", const=1000, type=int, metavar="N",
                        help="Опыт с моделью мира: исследовать маленькую "
                             "вселенную N шагами и сверить восстановленную "
                             "модель с её настоящими правилами")
    parser.add_argument("--series", action="store_true",
                        help="Как менялся результат по каждому вопросу при "
                             "изменении условий")
    parser.add_argument("--tried", action="store_true",
                        help="Что уже проверяли и чем это кончилось "
                             "(чтобы не повторять эксперимент заново)")
    parser.add_argument("--lichess-token", action="store_true",
                        dest="lichess_token",
                        help="Ввести токен lichess.org и сохранить его в "
                             "диспетчере учётных данных (ввод не виден)")
    parser.add_argument("--lichess", nargs="?", const=0, type=int, metavar="N",
                        help="Сыграть N партий на lichess.org с живой доской "
                             "и ходом размышлений; без числа — что доступно")
    parser.add_argument("--watch-port", type=int, default=0, dest="watch_port",
                        help="Порт окна наблюдения (только 127.0.0.1)")
    parser.add_argument("--practice", nargs="?", const=0, type=int, metavar="N",
                        help="Сыграть N партий на проверенном движке и добавить "
                             "их в корпус; без числа — показать накопленное")
    parser.add_argument("--forget-junk", action="store_true", dest="forget_junk",
                        help="Показать записи памяти, которые сегодняшние "
                             "ограждения не пропустили бы; удаляет только с --yes")
    parser.add_argument("--capabilities", action="store_true",
                        help="Что MANA умеет сверх поставки и доказано ли это")
    parser.add_argument("--acquire", metavar="ИМЯ",
                        help="Приобрести способность: показывает план; "
                             "устанавливает только вместе с --yes")
    parser.add_argument("--yes", action="store_true",
                        help="Согласие на установку для --acquire")
    parser.add_argument("--list-brains", action="store_true",
                        help="Показать все мозги: какие настроены, готовы, в кулдауне или исчерпали free-tier")
    parser.add_argument("--brains-status", action="store_true",
                        help="Полный JSON-статус пула мозгов (замеренные латентность и качество)")
    parser.add_argument("--brain-policy", default=None,
                        choices=["capability_first", "fastest", "cheapest", "least_loaded", "strongest", "round_robin"],
                        help="Как пул ранжирует мозги при выборе")
    parser.add_argument("--brains-file", default=None, metavar="PATH",
                        help="JSON-файл, дополняющий/переопределяющий каталог мозгов")
    parser.add_argument("--no-external-brains", action="store_true",
                        help="Только локальные мозги: ни один внешний API не вызывается")
    parser.add_argument("--allow-paid-brains", action="store_true",
                        help="Разрешить платные мозги (по умолчанию используются только бесплатные и локальные)")
    parser.add_argument("--ask", metavar="TASK", default=None,
                        help="Задать один вопрос и выйти (обычный путь solve_task)")
    parser.add_argument("--consensus", type=int, default=None, metavar="N",
                        help="Спросить N мозгов параллельно и показать ответ вместе со степенью их согласия")
    parser.add_argument("--decompose", metavar="TASK", default=None,
                        help="Разложить задачу на подзадачи, решить их на разных мозгах и собрать ответ")
    parser.add_argument("--list-code-targets", action="store_true", help="Показать список whitelisted целей для self-improve-code")
    parser.add_argument("--self-improve-code", metavar="TARGET_ID", default=None, help="Предложить и (при принятии gate'ом) применить патч кода для указанной цели")
    parser.add_argument("--code-instruction", default="", help="Инструкция для LLM при --self-improve-code")
    parser.add_argument("--code-history", nargs="?", const="__all__", default=None, metavar="TARGET_ID", help="Показать журнал применённых code-патчей (для всех целей или указанной)")
    parser.add_argument("--rollback-code", metavar="TARGET_ID", default=None, help="Откатить последний применённый патч для указанной цели")
    parser.add_argument("--graph-memory-status", action="store_true", help="Показать статистику графовой памяти (узлы/рёбра) текущей сессии")
    parser.add_argument("--graph-memory-search", metavar="QUERY", default=None, help="Найти контекст через обход графа памяти (не просто top-k)")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог эволюции и benchmark")
    parser.add_argument("--voice-model", default="small")
    parser.add_argument("--voice-seconds", type=float, default=30.0)
    parser.add_argument("--voice-language", default="ru")
    parser.add_argument("--voice-tts-backend", choices=["auto", "silero", "pyttsx3"], default="auto")
    parser.add_argument("--voice-speaker", default="xenia")
    parser.add_argument("--voice-output-device", type=int, default=None)
    parser.add_argument("--list-audio-devices", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # The console is this process's output device, so say so once. Library
    # code emits events (see mana/events.py) and no longer assumes a
    # terminal exists; without a sink installed those events go nowhere,
    # which is exactly what the windowed app wants and exactly what a CLI
    # run must not do.
    events.install_console_sink()

    if args.paths_status:
        print(json.dumps(paths.status(), ensure_ascii=False, indent=2))
        return 0

    # Before any agent is constructed: reading the record must not require
    # loading embedding models, and a journal that cannot be read without
    # starting the thing it observes is not evidence anybody will look at.
    if args.journal is not None:
        return _show_journal(int(args.journal))

    if args.findings is not None:
        return _show_findings(int(args.findings))

    if args.propose is not None:
        return _show_proposals(int(args.propose))

    if args.laws:
        return _show_laws()

    if args.next_probe:
        return _show_next()

    if args.cycle:
        return _run_cycle()

    if args.world is not None:
        return _run_world(int(args.world))

    if args.series:
        return _show_series()

    if args.tried:
        return _show_tried()

    if args.lichess_token:
        return _lichess_token()
    if args.lichess is not None:
        return _lichess(int(args.lichess), int(args.watch_port))
    if args.practice is not None:
        return _practice(int(args.practice))

    if args.forget_junk:
        return _forget_junk(bool(args.yes))

    if args.capabilities:
        return _show_capabilities()

    if args.acquire:
        return _acquire_capability(str(args.acquire), bool(args.yes))

    # Handled before any agent is constructed: reporting the version must
    # not require loading embedding models or opening databases.
    if args.version:
        print(format_version_report())
        return 0

    if args.list_audio_devices:
        if not HAS_SOUNDDEVICE:
            print("sounddevice не установлен"); return 1
        print(sd.query_devices()); return 0

    cfg = build_config(args)
    agent = ManaAgent(cfg)
    if args.learn_topic is not None:
        print(json.dumps(agent.acquire_topic_from_web(args.learn_topic, domain=args.learn_domain or args.learn_topic), ensure_ascii=False, indent=2)); return 0
    if args.learn is not None:
        print(json.dumps(agent.acquire_knowledge(args.learn, domain=args.learn_domain, recursive=not args.learn_nonrecursive), ensure_ascii=False, indent=2)); return 0
    if args.knowledge_status:
        print(json.dumps(agent.knowledge_status(), ensure_ascii=False, indent=2)); return 0
    if args.hardware_status:
        print(json.dumps(agent._json_safe(agent.hardware_status()), ensure_ascii=False, indent=2)); return 0
    if args.list_tools:
        print(json.dumps(agent.tools_status(), ensure_ascii=False, indent=2)); return 0
    if args.list_brains:
        print(format_brains(agent.brains_status())); return 0
    if args.brains_status:
        print(json.dumps(agent._json_safe(agent.brains_status()), ensure_ascii=False, indent=2)); return 0
    if args.decompose is not None:
        print(json.dumps(agent._json_safe(agent.solve_decomposed(args.decompose)),
                          ensure_ascii=False, indent=2)); return 0
    if args.consensus is not None and args.ask is not None:
        print(json.dumps(agent._json_safe(agent.ask_consensus(args.ask, n=int(args.consensus))),
                          ensure_ascii=False, indent=2)); return 0
    if args.ask is not None:
        print(json.dumps(agent._json_safe(agent.solve_task(args.ask)), ensure_ascii=False, indent=2)); return 0
    if args.list_code_targets:
        print(json.dumps(agent.list_code_targets(), ensure_ascii=False, indent=2)); return 0
    if args.code_history is not None:
        target_id = None if args.code_history == "__all__" else args.code_history
        print(json.dumps(agent.code_history(target_id), ensure_ascii=False, indent=2)); return 0
    if args.rollback_code is not None:
        print(json.dumps(agent.rollback_code(args.rollback_code), ensure_ascii=False, indent=2)); return 0
    if args.self_improve_code is not None:
        report = agent.self_improve_code(args.self_improve_code, args.code_instruction)
        print(json.dumps(agent._json_safe(report), ensure_ascii=False, indent=2)); return 0
    if args.graph_memory_status:
        print(json.dumps(agent.graph_memory_status(), ensure_ascii=False, indent=2)); return 0
    if args.graph_memory_search is not None:
        print(json.dumps(agent.graph_memory_search(args.graph_memory_search), ensure_ascii=False, indent=2)); return 0
    if args.memory_status:
        print(json.dumps({"version": agent.VERSION, "session_id": agent.session_id, "db": cfg.memory_db_path, "session": agent.persistent_memory.get_session(agent.session_id), "events": agent._memory_event_count(agent.session_id), "knowledge": agent.knowledge_status()}, ensure_ascii=False, indent=2)); return 0
    if args.memory_search is not None:
        print(json.dumps(agent._json_safe(agent.persistent_memory.safe_search_global(args.memory_search, cfg.memory_semantic_top_k)), ensure_ascii=False, indent=2))
        return 0
    if args.verify is not None:
        print(json.dumps(agent.tools.call("verify_arithmetic", expression=args.verify).output, ensure_ascii=False, indent=2)); return 0
    if args.run_code is not None:
        raw = str(args.run_code)
        path = Path(raw)
        if path.exists() and path.is_file():
            code = path.read_text(encoding="utf-8")
            result = agent.tools.call("run_code", code=code).output
            result["source"] = str(path.resolve())
        else:
            result = agent.tools.call("run_code", code=raw).output
            result["source"] = "inline"
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    if args.routing_benchmark:
        agent.routing_benchmark(); return 0
    if args.routing_holdout:
        print(json.dumps(agent.routing_holdout(), ensure_ascii=False, indent=2)); return 0
    if args.adaptive_benchmark:
        print(json.dumps(agent.adaptive_benchmark(False, repetitions=getattr(args, "adaptive_repetitions", None)), ensure_ascii=False, indent=2)); return 0
    if args.adaptive_holdout:
        print(json.dumps(agent.adaptive_benchmark(True, repetitions=getattr(args, "adaptive_holdout_repetitions", None)), ensure_ascii=False, indent=2)); return 0
    if args.benchmark:
        agent.benchmark(); return 0
    config_only_flags = (args.web_parallel or args.web_max_retries is not None or args.web_retry_delay is not None) and args.self_improve == 0 and not args.interactive and not args.voice and not args.routing_benchmark and not args.routing_holdout and not args.adaptive_benchmark and not args.adaptive_holdout and not args.run_code and args.verify is None and args.learn is None and args.learn_topic is None and not args.knowledge_status
    if config_only_flags:
        print(json.dumps({
            "version": agent.VERSION, "config_only": True,
            "web": {
                "enabled": bool(agent.web.enabled),
                "serialized_requests": bool(agent.config.web_serialized_requests),
                "max_retries": int(agent.config.web_max_retries),
                "retry_delay_seconds": float(agent.config.web_retry_delay_seconds),
            }
        }, ensure_ascii=False, indent=2)); return 0
    if args.self_improve > 0:
        agent.run_cycles(args.self_improve)
    elif args.interactive:
        agent.interactive()
    elif not args.voice:
        agent.run_cycles(args.cycles)

    if args.voice:
        try:
            VoiceInterface(agent, cfg).run()
        except Exception as exc:
            print(f"❌ Voice mode unavailable: {exc}")

    print("\nПровайдеры:")
    print(json.dumps(agent.llm.status(), ensure_ascii=False))
    print("\nИтог:")
    print(f"version={agent.VERSION}")
    print(f"cycle={agent.cycle}")
    print(f"best_pipeline_fitness={agent.best_pipeline_fitness:.3f}")
    print(f"pipeline={agent.pipeline.pretty()}")
    print(f"memory_entries={len(agent.memory.entries)}")
    print(f"persistent_memory_events={agent._memory_event_count(agent.session_id)}")
    print(f"experience_records={agent.experience.count()}")
    if agent.evolution_reports:
        r = agent.evolution_reports[-1]
        e = r["comparison"]["effect"]
        print("last_cycle_effect=")
        print(json.dumps({
            "cycle": r["cycle"], "verdict": r["verdict"], "accepted": r["accepted"],
            "train_quality": e["train_quality"], "holdout_quality": e["holdout_quality"],
            "train_p50_latency": e["train_p50_latency"], "train_p95_latency": e["train_p95_latency"],
            "train_timeout_rate": e["train_timeout_rate"], "train_fallback_rate": e["train_fallback_rate"],
            "decision": r.get("decision", {}),
            "pipeline_changes": r.get("pipeline_changes", {}),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
