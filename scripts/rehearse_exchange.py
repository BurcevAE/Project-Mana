#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/rehearse_exchange.py — прогнать обмен целиком, не вставая с места.

Зачем репетиция
---------------
Настоящая проверка требует второй машины, и это полдня: поставить, дать
мозги, дождаться цикла. Обидно потратить их и обнаружить, что изъян был в
самом протоколе. Здесь оба экземпляра поднимаются в отдельных процессах, с
отдельными каталогами состояния и отдельными `MANA_INSTANCE_ID`, — то есть
со **своими скрытыми выборками**, как на разных машинах.

Единственное, чего репетиция не даёт: настоящих экспериментов. Вердикты
здесь задаются сценарием, чтобы показать, как выглядит расхождение. На
второй машине их выносят ворота, и это единственная разница — но она и есть
то, ради чего всё построено.

    python scripts/rehearse_exchange.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PYTHON = sys.executable

#: Что делает каждая «машина». Запускается отдельным процессом: подменить
#: идентификатор внутри одного процесса значило бы проверить наличие
#: оператора if, а не разделение выборок.
ARM = r'''
import json, sys
sys.path.insert(0, r"{root}")
from mana.cognition import exchange
from mana.core import identity, splits

queue = exchange.Queue(r"{queue}")
out = {{"instance": identity.fingerprint(),
       "holdout": splits.HOLDOUT_V1.identity,
       "seed": splits.HOLDOUT_V1.effective_seed()}}

step = {step!r}

if step == "propose":
    # То, что в жизни запишет ResearchCycle сам.
    for name, steps in (("probe_then_answer", ["OBSERVE", "RETRIEVE", "ANSWER"]),
                        ("critique_twice", ["OBSERVE", "GENERATE", "CRITIQUE",
                                            "REPAIR", "ANSWER"])):
        record = queue.record_hypothesis(
            "create_program_template",
            {{"name": name, "steps": steps, "applicability": ["arithmetic"],
             "description": "порождено циклом; текст для человека"}})
        queue.record_report(record["hypothesis_id"], {verdict!r}, trials=40,
                            holdout=splits.HOLDOUT_V1.identity)
    written = exchange.export_bundle(r"{bundle}",
                                     queue.hypotheses(shareable_only=True),
                                     queue.reports())
    out.update(exported=written["hypotheses"], reports=written["reports"])

elif step == "receive":
    result = exchange.import_bundle(r"{bundle}", known=queue.known_ids())
    queue.absorb(result)
    from mana.cognition import genome as g
    local = g.CognitiveGenome(representations=g.baseline_representations(),
                              program_templates=g.baseline_templates(),
                              learning_rules=g.baseline_learning_rules())
    applied = []
    for h in result.hypotheses:
        try:
            exchange.to_local_proposal(h, local)
            applied.append(h.hypothesis_id)
        except Exception as exc:
            applied.append(None)
    # Свой вердикт, вынесенный ЗДЕСЬ. В жизни его дают ворота.
    for h in result.hypotheses:
        queue.record_report(h.hypothesis_id, {verdict!r}, trials=40,
                            holdout=splits.HOLDOUT_V1.identity)
    out.update(received=len(result.hypotheses), refused=len(result.refused),
               buildable=sum(1 for a in applied if a))
    written = exchange.export_bundle(r"{bundle_back}",
                                     queue.hypotheses(shareable_only=True),
                                     queue.reports())
    out.update(sent_back=written["reports"])

elif step == "judge":
    result = exchange.import_bundle(r"{bundle_back}", known=queue.known_ids())
    queue.absorb(result)
    out["consensus"] = {{k: [e["hypothesis_id"] for e in v]
                        for k, v in exchange.consensus(queue.reports()).items()}}
    out["replication"] = exchange.replication(queue.reports())

print("@@" + json.dumps(out, ensure_ascii=False, default=str))
'''


def run(root: Path, instance: str, data: Path, step: str, verdict: str,
        queue: Path, bundle: Path, bundle_back: Path) -> dict:
    env = dict(os.environ)
    env["MANA_INSTANCE_ID"] = instance
    env["MANA_DATA_DIR"] = str(data)
    env["PYTHONIOENCODING"] = "utf-8"
    code = ARM.format(root=root, queue=queue, bundle=bundle,
                      bundle_back=bundle_back, step=step, verdict=verdict)
    done = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    for line in done.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit(f"экземпляр {instance} не отчитался:\n"
                     f"{done.stdout}\n{done.stderr[-2000:]}")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="mana-rehearsal-"))
    bundle = work / "от-А.json"
    back = work / "от-Б.json"
    try:
        print("РЕПЕТИЦИЯ ОБМЕНА")
        print("=" * 62)
        print(f"рабочий каталог: {work}")
        print()

        print("1. Экземпляр А порождает гипотезы и принимает их у себя")
        a = run(REPO_ROOT, "rehearsal-alpha", work / "a", "propose", "ACCEPTED",
                work / "a" / "queue.json", bundle, back)
        print(f"   отпечаток        {a['instance']}")
        print(f"   скрытая выборка  {a['holdout']}  зерно {a['seed']}")
        print(f"   выгружено        {a['exported']} гипотез, {a['reports']} отчёта")
        print()

        print("2. Экземпляр Б загружает пакет и судит сам — и отклоняет")
        b = run(REPO_ROOT, "rehearsal-beta", work / "b", "receive", "REJECTED",
                work / "b" / "queue.json", bundle, back)
        print(f"   отпечаток        {b['instance']}")
        print(f"   скрытая выборка  {b['holdout']}  зерно {b['seed']}")
        print(f"   принято гипотез  {b['received']}, отклонено записей {b['refused']}")
        print(f"   построено на своём геноме: {b['buildable']} из {b['received']}")
        print()
        if a["seed"] == b["seed"]:
            print("   ! ВЫБОРКИ СОВПАЛИ — соль не работает, дальше смысла нет")
            return 1
        print("   выборки разошлись — значит согласие будет воспроизведением,")
        print("   а не следствием того, что оба тянули одни и те же задачи")
        print()

        print("3. Экземпляр А забирает ответ и смотрит, что вышло")
        c = run(REPO_ROOT, "rehearsal-alpha", work / "a", "judge", "ACCEPTED",
                work / "a" / "queue.json", bundle, back)
        groups = c["consensus"]
        print()
        for label, key in (("РАСХОЖДЕНИЯ", "divergent"),
                           ("подтверждено", "confirmed"),
                           ("опровергнуто", "refuted"),
                           ("не решено", "undecided")):
            ids = groups.get(key) or []
            print(f"   {label:14s} {len(ids)}" + (f"  {', '.join(ids)}" if ids else ""))

        print()
        if groups.get("divergent"):
            print("=" * 62)
            print("НАЙДЕНО РАСХОЖДЕНИЕ — ради этого всё и строилось.")
            print("Одна и та же гипотеза принята на одной установке и отклонена")
            print("на другой. Значит изменение УСЛОВНО, и условие — факт про")
            print("среды, которого ни один экземпляр в одиночку увидеть не мог.")
            print()
            print("Порядок действий такой же и на настоящей второй машине;")
            print("разница одна: вердикты там вынесут ворота, а не сценарий.")
            return 0

        print("расхождений нет — репетиция прошла, но ничего не показала")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
