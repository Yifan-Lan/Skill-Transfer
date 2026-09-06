#!/usr/bin/env python3
"""Convert raw InfiAgent-DABench (DAEval) into our pipeline layout.

Produces, under --out (default: this dir):
  dataset.json          list of tasks  {id, question, constraints, format,
                        file_name, level, concepts, type:"dabench",
                        task_path:"tables", golden_answers:[[name,val],...]}
  split_manifest.json   stratified-by-level splits (train/val/test + probe_50)
  tables/<csv>          the 52 shared CSV inputs (referenced by file_name)

Source (already downloaded by the caller into raw/):
  raw/da-dev-questions.jsonl   {id,question,concepts,constraints,format,file_name,level}
  raw/da-dev-labels.jsonl      {id, common_answers:[[name,value],...]}
  raw/da-dev-tables/<csv>

The golden_answers field is later stripped from each case's task_meta by
multi_helpers.cmd_prepare (added to the strip set), so the agent can't read it.
"""
from __future__ import annotations
import argparse, json, random, shutil, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "raw"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val",   type=int, default=30)
    ap.add_argument("--test",  type=int, default=150)
    ap.add_argument("--probe", type=int, default=50)
    args = ap.parse_args()

    raw = Path(args.raw); out = Path(args.out)
    questions = read_jsonl(raw / "da-dev-questions.jsonl")
    labels = {d["id"]: d["common_answers"] for d in read_jsonl(raw / "da-dev-labels.jsonl")}
    tables_src = raw / "da-dev-tables"

    tables_dst = out / "tables"; tables_dst.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    skipped = 0
    for q in questions:
        rid = q["id"]
        fname = q["file_name"]
        if rid not in labels:
            skipped += 1; continue
        if not (tables_src / fname).exists():
            print(f"WARN: table missing for id={rid}: {fname}", file=sys.stderr); skipped += 1; continue
        # copy shared CSV once
        dst = tables_dst / fname
        if not dst.exists():
            shutil.copy2(tables_src / fname, dst)
        tasks.append({
            "id": f"DA{rid:04d}",
            "question": q["question"],
            "constraints": q.get("constraints", ""),
            "format": q.get("format", ""),
            "concepts": q.get("concepts", []),
            "level": q.get("level", ""),
            "file_name": fname,
            "type": "dabench",
            "task_path": "tables",
            "golden_answers": labels[rid],
        })

    (out / "dataset.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── stratified-by-level split ────────────────────────────────────────────
    rng = random.Random(args.seed)
    by_level: dict[str, list[str]] = collections.defaultdict(list)
    for t in tasks:
        by_level[t["level"]].append(t["id"])
    for lv in by_level:
        rng.shuffle(by_level[lv])

    def take_stratified(n: int, pool: dict[str, list[str]]) -> list[str]:
        """Round-robin across levels so each split keeps the easy/med/hard mix."""
        picked: list[str] = []
        levels = list(pool.keys())
        while len(picked) < n and any(pool[lv] for lv in levels):
            for lv in levels:
                if len(picked) >= n:
                    break
                if pool[lv]:
                    picked.append(pool[lv].pop())
        return picked

    # deep-copy pools so successive splits are disjoint
    pool = {lv: list(ids) for lv, ids in by_level.items()}
    train = take_stratified(args.train, pool)
    val   = take_stratified(args.val, pool)
    test  = take_stratified(args.test, pool)
    # probe_N = a stratified subset drawn from the TEST split (for the cheap
    # strong-vs-weak gap probe; a subset of test on purpose).
    probe_pool: dict[str, list[str]] = collections.defaultdict(list)
    id2level = {t["id"]: t["level"] for t in tasks}
    for tid in test:
        probe_pool[id2level[tid]].append(tid)
    probe = take_stratified(args.probe, {lv: list(ids) for lv, ids in probe_pool.items()})

    manifest = {
        "seed": args.seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test), "probe": len(probe)},
        "ids": {
            f"train_{len(train)}": train,
            f"val_{len(val)}": val,
            f"test_{len(test)}": test,
            f"probe_{len(probe)}": probe,
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lv_counts = collections.Counter(t["level"] for t in tasks)
    print(f"tasks={len(tasks)} skipped={skipped} tables={len(list(tables_dst.glob('*.csv')))}")
    print(f"levels: {dict(lv_counts)}")
    print(f"splits: train={len(train)} val={len(val)} test={len(test)} probe={len(probe)}")
    print(f"split names: {list(manifest['ids'].keys())}")


if __name__ == "__main__":
    main()
