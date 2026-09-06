#!/usr/bin/env python3
"""Aggregate OfficeQA **within-tolerance accuracy** from existing eval JSONs.

OfficeQA eval already stores per-case `scores_by_tolerance`
({exact, 0.1pct, 1pct, 5pct}, each 1.0/0.0) in `eval.json` (pass@1) or
`eval_sample_NN.json` (pass@N).  This script just AGGREGATES those into
per-tolerance accuracy — no re-run, no API calls.

For a tolerance T, accuracy@T = fraction of cases whose answer is within T of
the gold value.  (exact == the headline pass@1.)

Output (per eval dir): `tolerance_summary.json`, plus a comparison table.

Usage:
    python compute_tolerance_accuracy.py \
        --eval-dirs runs/eval_baseline \
                    runs/eval_adapted
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TOLS = ["exact", "0.1pct", "1pct", "5pct"]


def _sample_files(eval_dir: Path) -> list[Path]:
    """Pass@N → eval_sample_NN.json; else single eval.json."""
    samples = sorted(eval_dir.glob("eval_sample_*.json"))
    if samples:
        return samples
    single = eval_dir / "eval.json"
    return [single] if single.exists() else []


def _accuracy_one(eval_json: Path) -> dict:
    """Per-tolerance accuracy + pass rate over the cases in one eval JSON."""
    d = json.loads(eval_json.read_text(encoding="utf-8"))
    n = len(d)
    n_with_tol = sum(1 for v in d.values() if v.get("scores_by_tolerance"))
    acc = {}
    for t in TOLS:
        vals = [float(v["scores_by_tolerance"].get(t, 0.0))
                for v in d.values() if v.get("scores_by_tolerance")]
        acc[t] = (sum(vals) / len(vals)) if vals else None
    pass_rate = (sum(1 for v in d.values() if v.get("pass")) / n) if n else None
    return {"n_cases": n, "n_with_tolerance": n_with_tol, "pass_rate": pass_rate, "accuracy": acc}


def score_eval_dir(eval_dir: Path) -> dict:
    files = _sample_files(eval_dir)
    if not files:
        raise SystemExit(f"ERROR: no eval.json / eval_sample_*.json in {eval_dir}")

    per_sample = [{"file": f.name, **_accuracy_one(f)} for f in files]
    n_samples = len(per_sample)

    # mean across samples, per tolerance (and pass rate)
    def _mean_over_samples(key_path):
        vals = []
        for s in per_sample:
            v = s
            for k in key_path:
                v = v[k] if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        return statistics.mean(vals) if vals else None

    mean_acc = {t: _mean_over_samples(["accuracy", t]) for t in TOLS}
    mean_pass = _mean_over_samples(["pass_rate"])

    # best-of-samples per case, per tolerance (only meaningful when n_samples > 1)
    best_acc = None
    if n_samples > 1:
        # union over samples: a case counts at tolerance T if ANY sample hit T
        by_case: dict[str, dict[str, float]] = {}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            for cid, v in d.items():
                sbt = v.get("scores_by_tolerance") or {}
                slot = by_case.setdefault(cid, {t: 0.0 for t in TOLS})
                for t in TOLS:
                    slot[t] = max(slot[t], float(sbt.get(t, 0.0)))
        best_acc = {t: (statistics.mean(c[t] for c in by_case.values()) if by_case else None)
                    for t in TOLS}

    summary = {
        "eval_dir": str(eval_dir),
        "n_samples": n_samples,
        "n_cases": per_sample[0]["n_cases"],
        "pass_rate_mean": mean_pass,
        "accuracy_within_tolerance_mean": mean_acc,
        "accuracy_within_tolerance_best_of_samples": best_acc,
        "per_sample": per_sample,
    }
    (eval_dir / "tolerance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dirs", nargs="+", required=True,
                   help="One or more OfficeQA eval dirs (each with eval.json or eval_sample_NN.json)")
    args = p.parse_args()

    rows = []
    for ed in args.eval_dirs:
        ed = Path(ed).resolve()
        s = score_eval_dir(ed)
        rows.append((ed.name, s))

    print("\n" + "═" * 78)
    print("  OfficeQA ACCURACY WITHIN TOLERANCE (mean over samples)")
    print("═" * 78)
    hdr = f"  {'eval dir':<22s} {'cases':>5s} {'exact':>8s} {'≤0.1%':>8s} {'≤1%':>8s} {'≤5%':>8s}"
    print(hdr)
    print(f"  {'-'*22:<22s} {'-'*5:>5s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s}")
    for name, s in rows:
        a = s["accuracy_within_tolerance_mean"]
        def f(x):
            return f"{x*100:>6.1f}%" if x is not None else "   n/a"
        print(f"  {name:<22s} {s['n_cases']:>5d} {f(a['exact']):>8s} "
              f"{f(a['0.1pct']):>8s} {f(a['1pct']):>8s} {f(a['5pct']):>8s}")
    # best-of-samples block, only if any arm had >1 sample
    if any(s["accuracy_within_tolerance_best_of_samples"] for _, s in rows):
        print(f"\n  best-of-{max(s['n_samples'] for _,s in rows)}-samples (union):")
        for name, s in rows:
            b = s["accuracy_within_tolerance_best_of_samples"]
            if not b:
                continue
            def f(x):
                return f"{x*100:>6.1f}%" if x is not None else "   n/a"
            print(f"  {name:<22s} {s['n_cases']:>5d} {f(b['exact']):>8s} "
                  f"{f(b['0.1pct']):>8s} {f(b['1pct']):>8s} {f(b['5pct']):>8s}")
    print("\n  'exact' == headline pass@1.  Each tolerance is cumulative (≤5% includes ≤1% includes exact).")
    print("  files : <eval-dir>/tolerance_summary.json\n")


if __name__ == "__main__":
    main()
