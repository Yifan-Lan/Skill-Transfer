#!/usr/bin/env python3
"""Post-hoc SpreadsheetBench **cell match ratio** for one or more eval dirs.

The official SB eval (`run_eval.sh` → `pipeline_helpers._eval_one` →
`evaluation.compare_workbooks`) is BINARY and short-circuits on the first
mismatched answer cell, so the eval JSONs only store `{"pass", "message"}` —
no cell-level ratio.  This script recovers a finer-grained **cell match ratio**
(fraction of answer-range cells that match ground truth) by RE-SCORING the
already-saved `output.xlsx` files.  It does NOT re-run the agent — it only reads
files already on disk, so it is local and free.

It reuses `evaluation.py`'s exact cell-equality (`compare_cell_value`,
`transform_value`) and range parsing (`generate_cell_names`) so a case with
ratio == 1.0 is exactly the set of cases the binary metric marks `pass` (a
built-in cross-check is printed).

Output (per eval dir): `cell_match_summary.json` + `cell_match_per_case.json`,
plus a comparison table across all eval dirs.

Usage:
    python compute_cell_match.py \
        --eval-dirs runs/eval_adapted \
        --dataset-dir data/spreadsheetbench/test_200

Multiple arms in one shot (shared --dataset-dir):
    python compute_cell_match.py \
        --eval-dirs runs/eval_baseline runs/eval_adapted \
        --dataset-dir data/spreadsheetbench/test_200
"""
from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# SpreadsheetBench's own evaluator is reused so that cell_match_ratio == 1.0 is
# exactly the benchmark's binary pass.  Clone the benchmark next to this repo,
# or point SPREADSHEETBENCH_DIR at your checkout.
_sb = os.environ.get("SPREADSHEETBENCH_DIR") or (REPO_ROOT.parent / "SpreadsheetBench")
sys.path.insert(0, str(Path(_sb) / "evaluation"))
import openpyxl  # noqa: E402
# Reuse the OFFICIAL equality + range logic so ratio==1.0 ⟺ binary pass.
from evaluation import compare_cell_value, generate_cell_names  # noqa: E402


def _answer_cells(answer_position: str, first_sheet: str) -> list[tuple[str, str]]:
    """Parse 'Sheet!A1:A3,Sheet2!B1' → [(sheet, cell), ...].

    Mirrors evaluation.compare_workbooks parsing EXACTLY (quote-strip only,
    NO whitespace strip) so the result stays consistent with the official
    binary metric — e.g. a leading non-breaking space in the sheet name
    ("\\xa0'Sheet1'") must remain so the sheet is correctly "not found",
    matching the official fail (do NOT .strip() it away)."""
    out: list[tuple[str, str]] = []
    for scr in answer_position.split(","):
        if "!" in scr:
            sheet, rng = scr.split("!", 1)
            sheet = sheet.lstrip("'").rstrip("'")
        else:
            sheet, rng = first_sheet, scr
        sheet = sheet.lstrip("'").rstrip("'")
        rng = rng.lstrip("'").rstrip("'")
        for cell in generate_cell_names(rng):
            out.append((sheet, cell))
    return out


def cell_match(gt_file: Path, proc_file: Path, answer_position: str) -> dict:
    """Return {matched, total, ratio, exact, msg} for one (gt, proc) pair.

    Robust to the same inputs the official eval chokes on (e.g. whole-column
    'A:A' answer ranges): such cases raise in range parsing — the official
    `_eval_one` catches that and marks the case fail, so here we return
    ratio=None (unscorable; excluded from the cell-match mean, reported as
    skipped) rather than crashing the whole run.
    """
    try:
        wb_gt = openpyxl.load_workbook(filename=str(gt_file), data_only=True)
    except Exception as e:  # noqa: BLE001
        return {"matched": 0, "total": 0, "ratio": None, "exact": False,
                "msg": f"gt load error: {e}"}

    try:
        cells = _answer_cells(answer_position, wb_gt.sheetnames[0])
    except Exception as e:  # noqa: BLE001
        return {"matched": 0, "total": 0, "ratio": None, "exact": False,
                "msg": f"unparseable answer_position {answer_position!r}: {e}"}
    total = len(cells)
    if total == 0:
        return {"matched": 0, "total": 0, "ratio": None, "exact": False,
                "msg": "empty answer_position"}

    if not proc_file.exists():
        return {"matched": 0, "total": total, "ratio": 0.0, "exact": False,
                "msg": "output.xlsx not found"}
    try:
        wb_proc = openpyxl.load_workbook(filename=str(proc_file), data_only=True)
    except Exception as e:  # noqa: BLE001
        return {"matched": 0, "total": total, "ratio": 0.0, "exact": False,
                "msg": f"proc load error: {e}"}

    matched = 0
    try:
        for sheet, cell in cells:
            if sheet not in wb_gt.sheetnames:
                continue  # answer sheet absent in GT — count as mismatch
            if sheet not in wb_proc.sheetnames:
                continue  # missing worksheet in output → all those cells mismatch
            gt_c = wb_gt[sheet][cell]
            proc_c = wb_proc[sheet][cell]
            # A bare column/row token (e.g. 'C', '4') resolves to a tuple of
            # cells, not a single cell — the official eval hits .value on the
            # tuple and the case is caught-as-fail. Mirror that: unscorable.
            if not (hasattr(gt_c, "value") and hasattr(proc_c, "value")):
                raise TypeError(f"cell {cell!r} is not a single cell")
            if compare_cell_value(gt_c.value, proc_c.value):
                matched += 1
    except Exception as e:  # noqa: BLE001
        return {"matched": 0, "total": total, "ratio": None, "exact": False,
                "msg": f"cell-access error ({e}); official marks this case fail"}
    ratio = matched / total
    return {"matched": matched, "total": total, "ratio": ratio,
            "exact": (matched == total), "msg": ""}


def _resolve_golden(dataset_dir: Path, task_meta: dict) -> Path | None:
    src = dataset_dir / task_meta.get("spreadsheet_path", "")
    tid = str(task_meta.get("id", ""))
    for pat in (f"*_{tid}_golden.xlsx", "golden.xlsx", f"*_{tid}_answer.xlsx"):
        hits = sorted(src.glob(pat))
        if hits:
            return hits[0]
    return None


def _list_samples(case_dir: Path) -> list[int]:
    iters = []
    for d in sorted(case_dir.glob("iter_*")):
        if (d / "weak_outputs" / "output.xlsx").exists() or d.is_dir():
            try:
                iters.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(set(iters))


def score_eval_dir(eval_dir: Path, dataset_dir: Path) -> dict:
    cases_root = eval_dir / "cases"
    if not cases_root.is_dir():
        raise SystemExit(f"ERROR: {cases_root} not found — is --eval-dirs correct?")

    per_case: dict[str, dict] = {}
    skipped: list[str] = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        tm_file = case_dir / "task_meta.json"
        if not tm_file.exists():
            skipped.append(f"{cid}: no task_meta.json")
            continue
        tm = json.loads(tm_file.read_text(encoding="utf-8"))
        if tm.get("type") == "officeqa":
            skipped.append(f"{cid}: officeqa (not a spreadsheet task)")
            continue
        ans_pos = tm.get("answer_position", "")
        golden = _resolve_golden(dataset_dir, tm)
        if golden is None:
            skipped.append(f"{cid}: golden not found")
            continue

        sample_ratios: list[float] = []
        sample_detail: list[dict] = []
        for it in _list_samples(case_dir):
            proc = case_dir / f"iter_{it:02d}" / "weak_outputs" / "output.xlsx"
            r = cell_match(golden, proc, ans_pos)
            sample_detail.append({"iter": it, **r})
            if r["ratio"] is not None:
                sample_ratios.append(r["ratio"])
        if not sample_ratios:
            skipped.append(f"{cid}: no scorable samples")
            continue
        per_case[cid] = {
            "mean_ratio": statistics.mean(sample_ratios),
            "best_ratio": max(sample_ratios),
            "n_samples": len(sample_ratios),
            "any_exact": any(d["exact"] for d in sample_detail),
            "samples": sample_detail,
        }

    # ── Aggregate ────────────────────────────────────────────────────────────
    case_means = [c["mean_ratio"] for c in per_case.values()]
    case_bests = [c["best_ratio"] for c in per_case.values()]
    # per-sample-index dataset mean (each iter = one independent run)
    by_iter: dict[int, list[float]] = {}
    for c in per_case.values():
        for s in c["samples"]:
            if s["ratio"] is not None:
                by_iter.setdefault(s["iter"], []).append(s["ratio"])
    per_run = {f"iter_{k:02d}": {"n_cases": len(v),
                                 "cell_match_mean": statistics.mean(v),
                                 "cell_match_std": statistics.stdev(v) if len(v) > 1 else 0.0}
               for k, v in sorted(by_iter.items())}

    n_exact_best = sum(1 for c in per_case.values() if c["best_ratio"] == 1.0)
    summary = {
        "eval_dir": str(eval_dir),
        "dataset_dir": str(dataset_dir),
        "n_cases": len(per_case),
        "n_skipped": len(skipped),
        # primary headline metrics (averaged over cases)
        "cell_match_mean_of_case_means": statistics.mean(case_means) if case_means else None,
        "cell_match_mean_of_case_bests": statistics.mean(case_bests) if case_bests else None,
        # cross-check vs the binary metric: #cases with a fully-correct sample
        "exact_pass_at_k_count": n_exact_best,
        "exact_pass_at_k_rate": (n_exact_best / len(per_case)) if per_case else None,
        "per_run": per_run,
        "skipped": skipped[:30],
    }
    (eval_dir / "cell_match_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (eval_dir / "cell_match_per_case.json").write_text(
        json.dumps(per_case, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dirs", nargs="+", required=True,
                   help="One or more SB ablation eval dirs (cases/<id>/iter_NN/weak_outputs/output.xlsx)")
    p.add_argument("--dataset-dir", required=True,
                   help="SB dataset dir with golden files (e.g. .../test_200)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    def _pct(x):  # None-safe percentage formatter
        return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "  n/a"

    rows = []
    for ed in args.eval_dirs:
        ed = Path(ed).resolve()
        print(f"[scoring] {ed.name} …", flush=True)
        s = score_eval_dir(ed, dataset_dir)
        rows.append((ed.name, s))
        if s["n_cases"] == 0:
            print(f"  n_cases=0 skipped={s['n_skipped']} — no scorable SpreadsheetBench cases "
                  f"(not a cell-range dataset? e.g. WTQ/OfficeQA). Skipping.", flush=True)
            continue
        print(f"  n_cases={s['n_cases']} skipped={s['n_skipped']}  "
              f"cell_match(mean-of-case-means)={_pct(s['cell_match_mean_of_case_means'])}  "
              f"exact_pass_rate={_pct(s['exact_pass_at_k_rate'])}", flush=True)

    print("\n" + "═" * 86)
    print("  SpreadsheetBench CELL MATCH RATIO (post-hoc, over saved output.xlsx)")
    print("═" * 86)
    print(f"  {'eval dir':<46s} {'cases':>5s} {'cm_mean':>8s} {'cm_best':>8s} {'exact@k':>8s}")
    print(f"  {'-'*46:<46s} {'-'*5:>5s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s}")
    for name, s in rows:
        print(f"  {name:<46s} {s['n_cases']:>5d} "
              f"{_pct(s['cell_match_mean_of_case_means']):>8s} "
              f"{_pct(s['cell_match_mean_of_case_bests']):>8s} "
              f"{_pct(s['exact_pass_at_k_rate']):>8s}")
    print("\n  cm_mean = avg over cases of (mean cell-match ratio across that case's samples)")
    print("  cm_best = avg over cases of (best sample's cell-match ratio)")
    print("  exact@k = fraction of cases with ≥1 fully-correct sample (== binary pass@k cross-check)")
    print("  files   : <eval-dir>/cell_match_summary.json + cell_match_per_case.json\n")


if __name__ == "__main__":
    main()
