"""Preprocess WikiTableQuestions into a SpreadsheetBench-shaped dataset.

Inputs (cloned in advance):
  data/benchmarks/wtq/raw/data/pristine-unseen-tables.tsv  (test split, 4344 rows)
  data/benchmarks/wtq/raw/csv/<series>/<table>.csv          (per-table CSVs)

Outputs:
  data/benchmarks/wtq/dataset.json                                          (mirror of SSBench schema)
  data/benchmarks/wtq/spreadsheet/<id>/1_<id>_input.xlsx                    (CSV → xlsx)
  data/benchmarks/wtq/splits/wtq_test_n<n>_seed<seed>.json                  (which 70 ids to use)

Selection protocol (reproducible):
  1. Load pristine-unseen-tables.tsv
  2. Filter rows whose targetValue contains '|' (multi-answer cases)
  3. Filter rows whose csv path doesn't exist on disk
  4. Apply unescape (\\n, \\p, \\\\) to the question (utterance) and answer (targetValue)
  5. Shuffle deterministically with --seed
  6. Take the first --n entries

CSV → XLSX conversion:
  - Sheet1: full table from the CSV (header row + data rows)
  - Answer: empty worksheet; executor writes its answer string to Answer!A1

Usage:
  python -m data.prepare_wtq                              # defaults: n=70, seed=0
  python -m data.prepare_wtq --n 100 --seed 1
  python -m data.prepare_wtq --force                      # re-generate even if outputs exist
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook


def tsv_unescape(x: str) -> str:
    """Unescape strings from the WTQ TSV (verbatim from raw/evaluator.py).

    Inlined to avoid importing raw/evaluator.py (which carries py2-style
    `print` statements). Escapes: `\\n`→newline, `\\p`→`|`, `\\\\`→`\\`.
    """
    return x.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


# Resolve relative to THIS file so the script works from any CWD and matches
# the actual layout (data_wkt/benchmarks/wtq), not the original repo's data/.
WTQ_ROOT = Path(__file__).resolve().parent / "benchmarks" / "wtq"
RAW_DIR = WTQ_ROOT / "raw"
TSV_PATH = RAW_DIR / "data" / "pristine-unseen-tables.tsv"


def load_test_rows() -> list[dict]:
    """Read pristine-unseen-tables.tsv → list of {id, question, csv, target_raw}.

    The TSV's header line is `id\\tutterance\\tcontext\\ttargetValue`.
    """
    rows = []
    with open(TSV_PATH, encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append({
                "id": parts[idx["id"]],
                "question": tsv_unescape(parts[idx["utterance"]]),
                "csv": parts[idx["context"]],
                "target_raw": parts[idx["targetValue"]],   # keep escapes, unescape per item
            })
    return rows


def csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> dict:
    """Convert a WTQ CSV to an xlsx with Sheet1 = table, Answer = empty.

    Returns metadata about the conversion (n_rows, n_cols, truncated).
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    n_rows = 0
    n_cols = 0
    # Use utf-8-sig to silently swallow any BOM; errors=replace for the rare odd byte.
    # WTQ uses non-standard CSV escaping (per raw/table-to-csv.py:simple_normalize_text):
    #   literal `\`  → `\\`   in file
    #   literal `"`  → `\"`   in file
    # Standard csv.reader doesn't understand this; use escapechar + doublequote=False.
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, escapechar="\\", doublequote=False)
        for row in reader:
            ws.append(row)
            n_rows += 1
            n_cols = max(n_cols, len(row))

    # Add an empty Answer sheet — the executor will write its final answer to A1.
    wb.create_sheet("Answer")

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    return {"n_rows": n_rows, "n_cols": n_cols}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WTQ test split into SSBench-shaped xlsx dataset."
    )
    parser.add_argument("--n", type=int, default=70,
                        help="Number of test cases to sample.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for shuffling.")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate xlsx files even if they exist.")
    args = parser.parse_args()

    if not TSV_PATH.exists():
        sys.exit(
            f"ERROR: {TSV_PATH} not found.\n"
            f"Clone WTQ first:\n"
            f"  git clone --depth 1 https://github.com/ppasupat/WikiTableQuestions.git "
            f"{RAW_DIR}"
        )

    print(f"[prepare_wtq] Loading test split from {TSV_PATH}")
    all_rows = load_test_rows()
    print(f"  total: {len(all_rows)}")

    # Filter 1: drop multi-answer cases (targetValue contains pipe).
    single = [r for r in all_rows if "|" not in r["target_raw"]]
    print(f"  after dropping multi-answer (|): {len(single)}")

    # Filter 2: drop rows whose CSV is missing on disk.
    valid = []
    missing = 0
    for r in single:
        if (RAW_DIR / r["csv"]).exists():
            valid.append(r)
        else:
            missing += 1
    print(f"  after dropping missing CSVs: {len(valid)}  (dropped {missing})")

    # Filter 3: cap at requested n.
    if args.n > len(valid):
        sys.exit(f"ERROR: requested n={args.n} but only {len(valid)} usable rows remain.")
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    selected = valid[:args.n]
    print(f"  selected: {len(selected)} (seed={args.seed})")

    # Convert each CSV → xlsx.
    spreadsheet_root = WTQ_ROOT / "spreadsheet"
    spreadsheet_root.mkdir(parents=True, exist_ok=True)
    dataset_entries = []
    huge_tables = []
    for i, r in enumerate(selected):
        ex_id = r["id"]
        csv_src = RAW_DIR / r["csv"]
        xlsx_dest = spreadsheet_root / ex_id / f"1_{ex_id}_input.xlsx"

        if args.force or not xlsx_dest.exists():
            meta = csv_to_xlsx(csv_src, xlsx_dest)
        else:
            # Reuse existing xlsx; recompute n_rows by reopening for the dataset record.
            from openpyxl import load_workbook
            wb = load_workbook(xlsx_dest)
            ws = wb["Sheet1"]
            meta = {"n_rows": ws.max_row, "n_cols": ws.max_column}

        if meta["n_rows"] > 200:
            huge_tables.append((ex_id, meta["n_rows"]))

        # Unescape targetValue → single answer string (filter step 1 guarantees no |).
        golden = tsv_unescape(r["target_raw"])

        dataset_entries.append({
            "id": ex_id,
            "instruction": r["question"],
            "spreadsheet_path": f"spreadsheet/{ex_id}",
            "instruction_type": "WikiTableQuestion",
            "answer_position": "Answer!A1",
            "answer_sheet": "Answer",
            "golden_answer": golden,
            "source": {
                "csv_path": r["csv"],
                "n_rows": meta["n_rows"],
                "n_cols": meta["n_cols"],
            },
        })

        if (i + 1) % 20 == 0:
            print(f"  converted {i+1}/{len(selected)} CSVs")

    print(f"  converted {len(selected)}/{len(selected)} CSVs")
    if huge_tables:
        print(f"  WARN: {len(huge_tables)} table(s) > 200 rows (will use more tokens):")
        for ex_id, n in huge_tables[:5]:
            print(f"    {ex_id}: {n} rows")

    # Write dataset.json
    dataset_path = WTQ_ROOT / "dataset.json"
    dataset_path.write_text(
        json.dumps(dataset_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  wrote {dataset_path} ({len(dataset_entries)} entries)")

    # Write split.json with reproducibility metadata
    splits_dir = WTQ_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"wtq_test_n{args.n}_seed{args.seed}.json"
    split_info = {
        "split_name": "wtq_test",
        "source_tsv": str(TSV_PATH.relative_to(WTQ_ROOT.parent.parent)),
        "sample_seed": args.seed,
        "n_total_in_test": len(all_rows),
        "n_after_single_answer_filter": len(single),
        "n_after_csv_existence_filter": len(valid),
        "n_selected": len(selected),
        "skip_multi_answer": True,
        "test_ids": [r["id"] for r in selected],
        "test_indices": list(range(len(selected))),  # indices into dataset.json
    }
    split_path.write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  wrote {split_path}")

    print(f"\n[prepare_wtq] Done. {len(selected)} examples ready at:")
    print(f"  - {dataset_path}")
    print(f"  - {spreadsheet_root}/<id>/1_<id>_input.xlsx")
    print(f"  - {split_path}")


if __name__ == "__main__":
    main()
