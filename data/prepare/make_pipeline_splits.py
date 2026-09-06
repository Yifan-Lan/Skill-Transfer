#!/usr/bin/env python3
"""Create physical split dirs for the distillation pipeline: train_20 / val_30 / test_150.

Pipeline convention (same as officeqa_data/train_20 etc.): each split is a directory with
its own dataset.json (the split's SOURCE dataset — golden_answers kept here; prepare strips
them from per-case task_meta) plus a relative `tables` symlink to the shared CSVs.

- train_20: stratified-by-level (seeded) subset of the existing manifest train_60
- val_30 / test_150: exactly the existing manifest splits (test_150 unchanged)
- pairwise disjointness is ASSERTED, not assumed.
"""
from __future__ import annotations
import argparse, collections, json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-n", type=int, default=20)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads((HERE / "dataset.json").read_text())}
    man = json.loads((HERE / "split_manifest.json").read_text())
    train60, val30, test150 = (man["ids"][k] for k in ("train_60", "val_30", "test_150"))

    # stratified train_20 from train_60 (round-robin across levels, seeded)
    rng = random.Random(a.seed)
    by: dict[str, list] = collections.defaultdict(list)
    for i in train60:
        by[tasks[i]["level"]].append(i)
    for lv in by:
        rng.shuffle(by[lv])
    picked: list[str] = []
    while len(picked) < a.train_n and any(by.values()):
        for lv in list(by):
            if len(picked) >= a.train_n:
                break
            if by[lv]:
                picked.append(by[lv].pop())

    splits = {"train_20": picked, "val_30": val30, "test_150": test150}

    sets = {k: set(v) for k, v in splits.items()}
    for k1 in sets:
        for k2 in sets:
            if k1 < k2:
                ov = sets[k1] & sets[k2]
                assert not ov, f"OVERLAP {k1} ∩ {k2}: {sorted(ov)[:5]}"

    for name, ids in splits.items():
        d = HERE / name
        d.mkdir(exist_ok=True)
        subset = [tasks[i] for i in ids]
        (d / "dataset.json").write_text(json.dumps(subset, indent=2, ensure_ascii=False))
        link = d / "tables"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise SystemExit(f"{link} exists and is not a symlink — refusing to overwrite")
        os.symlink("../tables", link)  # RELATIVE (portability lesson from the Linux migration)
        lv = collections.Counter(t["level"] for t in subset)
        print(f"{name}: {len(subset)} tasks {dict(lv)}")

    man["ids"]["train_20"] = picked
    man.setdefault("notes", {})["pipeline_splits"] = (
        f"physical dirs train_20/val_30/test_150; train_20 ⊂ train_60 stratified seed={a.seed}; "
        "pairwise disjoint (asserted)"
    )
    (HERE / "split_manifest.json").write_text(json.dumps(man, indent=2))
    print("split_manifest.json updated (train_20 recorded)")


if __name__ == "__main__":
    main()
