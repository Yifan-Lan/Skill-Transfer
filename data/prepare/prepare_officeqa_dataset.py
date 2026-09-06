#!/usr/bin/env python3
"""
Prepare the OfficeQA dataset in the same directory structure as SpreadsheetBench.

Output layout (mirrors SpreadsheetBench):

  <out-dir>/
    dataset.json             — full task list (train + test)
    split_manifest.json      — {ids: {train: [...], test: [...]}}
    tasks/                   — analogous to SpreadsheetBench's spreadsheet/
      <uid>/
        prompt.txt           — the question (raw, no output instruction)
        golden_answer.txt    — expected answer
        <source>.txt         — source document(s)
    train_20/
      dataset.json           — train-split task list
      tasks/                 — per-case subdirs for train cases
    test_200/
      dataset.json           — test-split task list
      tasks/                 — per-case subdirs for test cases
    val_20/
      dataset.json           — val-split task list
      tasks/                 — per-case subdirs for val cases

dataset.json fields per task:
  id, question, answer, source_files, difficulty, type="officeqa",
  task_path="tasks/<uid>"      ← analogous to spreadsheet_path

Usage:
    python3.11 prepare_officeqa_dataset.py \
        --csv    officeqa/officeqa_full.csv \
        --zip    officeqa/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip \
        --out    officeqa_data \
        --train  20 \
        --val    20 \
        --test   200 \
        --seed   42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path


def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_files = [s.strip() for s in row["source_files"].split("\n") if s.strip()]
            rows.append({
                "id":           row["uid"].strip(),
                "question":     row["question"].strip(),
                "answer":       row["answer"].strip(),
                "source_files": source_files,
                "difficulty":   row["difficulty"].strip(),
                "type":         "officeqa",
            })
    return rows


def extract_txts(zip_path: Path, out_dir: Path) -> dict[str, bytes]:
    """Extract all txt files from zip; return {filename: bytes}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            fname = Path(name).name
            mapping[fname] = z.read(name)
    return mapping


def write_task_dir(task: dict, task_dir: Path, txt_data: dict[str, bytes]) -> None:
    """Write prompt.txt, golden_answer.txt, and source txts into task_dir."""
    task_dir.mkdir(parents=True, exist_ok=True)

    # INSTRUCTION.md — raw question only (no output instruction, matching SpreadsheetBench convention)
    (task_dir / "INSTRUCTION.md").write_text(task["question"], encoding="utf-8")

    # golden answer
    (task_dir / "golden_answer.txt").write_text(task["answer"], encoding="utf-8")

    # source txt files
    for fname in task["source_files"]:
        if fname in txt_data:
            dst = task_dir / fname
            if not dst.exists():
                dst.write_bytes(txt_data[fname])
        else:
            print(f"  WARN [{task['id']}]: source file not found in zip: {fname}", file=sys.stderr)


def write_split_dir(split_name: str, tasks: list[dict], root_tasks_dir: Path,
                    split_dir: Path) -> None:
    """Create <split_dir>/dataset.json and <split_dir>/tasks/ with per-case symlinks."""
    split_dir.mkdir(parents=True, exist_ok=True)
    split_tasks_dir = split_dir / "tasks"
    split_tasks_dir.mkdir(exist_ok=True)

    # dataset.json for this split (task_path rewritten to local tasks/)
    split_tasks = []
    for t in tasks:
        entry = dict(t)
        entry["task_path"] = f"tasks/{t['id']}"
        split_tasks.append(entry)
    (split_dir / "dataset.json").write_text(
        json.dumps(split_tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Copy (or symlink) task dirs into split/tasks/
    for t in tasks:
        src = root_tasks_dir / t["id"]
        dst = split_tasks_dir / t["id"]
        if not dst.exists():
            shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",   required=True, help="Path to officeqa_full.csv")
    parser.add_argument("--zip",   required=True, help="Path to treasury_bulletins_transformed.zip")
    parser.add_argument("--out",   required=True, help="Output dataset directory")
    parser.add_argument("--train", type=int, default=20,  help="Number of training cases")
    parser.add_argument("--val",   type=int, default=20,  help="Number of validation cases")
    parser.add_argument("--test",  type=int, default=200, help="Number of test cases")
    parser.add_argument("--seed",  type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    zip_path = Path(args.zip).resolve()
    out_dir  = Path(args.out).resolve()

    for p, name in [(csv_path, "CSV"), (zip_path, "ZIP")]:
        if not p.exists():
            sys.exit(f"ERROR: {name} not found: {p}")

    print(f"Loading CSV: {csv_path}")
    tasks = load_csv(csv_path)
    print(f"  {len(tasks)} tasks loaded")

    total_needed = args.train + args.val + args.test
    if len(tasks) < total_needed:
        sys.exit(f"ERROR: only {len(tasks)} tasks available, need {total_needed}")

    rng = random.Random(args.seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    train_tasks = shuffled[:args.train]
    val_tasks   = shuffled[args.train:args.train + args.val]
    test_tasks  = shuffled[args.train + args.val:args.train + args.val + args.test]
    print(f"  Split: {len(train_tasks)} train, {len(val_tasks)} val, {len(test_tasks)} test  (seed={args.seed})")

    print(f"Extracting txt files from zip…")
    txt_data = extract_txts(zip_path, out_dir / "_tmp_txt")
    shutil.rmtree(out_dir / "_tmp_txt", ignore_errors=True)
    print(f"  {len(txt_data)} txt files loaded")

    # Write root tasks/ directory
    tasks_dir = out_dir / "tasks"
    all_tasks = train_tasks + val_tasks + test_tasks
    print(f"Writing {len(all_tasks)} task directories → {tasks_dir}")
    for task in all_tasks:
        write_task_dir(task, tasks_dir / task["id"], txt_data)

    # Root dataset.json (task_path points to tasks/<uid>)
    root_tasks = []
    for t in all_tasks:
        entry = dict(t)
        entry["task_path"] = f"tasks/{t['id']}"
        root_tasks.append(entry)
    (out_dir / "dataset.json").write_text(
        json.dumps(root_tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"dataset.json → {out_dir / 'dataset.json'}")

    # Root split_manifest.json
    split_name_train = f"train_{args.train}"
    split_name_val   = f"val_{args.val}"
    split_name_test  = f"test_{args.test}"
    manifest = {
        "seed": args.seed,
        "counts": {"train": args.train, "val": args.val, "test": args.test},
        "ids": {
            split_name_train: [t["id"] for t in train_tasks],
            split_name_val:   [t["id"] for t in val_tasks],
            split_name_test:  [t["id"] for t in test_tasks],
        },
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"split_manifest.json → {out_dir / 'split_manifest.json'}")

    # train_N/, val_N/, and test_N/ split subdirs
    for split_name, split_tasks in [
        (split_name_train, train_tasks),
        (split_name_val,   val_tasks),
        (split_name_test,  test_tasks),
    ]:
        split_dir = out_dir / split_name
        print(f"Writing split subdir: {split_dir}")
        write_split_dir(split_name, split_tasks, tasks_dir, split_dir)

    print(f"\nDone.  Dataset ready at: {out_dir}")
    print(f"  Train : {out_dir / split_name_train}")
    print(f"  Val   : {out_dir / split_name_val}")
    print(f"  Test  : {out_dir / split_name_test}")


if __name__ == "__main__":
    main()
