#!/usr/bin/env python3
"""Break WikiTableQuestions results down by answer type.

Reports pass / wrong-value / no-output counts and accuracy split by number,
date, and string answers, using the official WTQ denotation matcher.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import wtq_eval  # noqa: E402

TYPES = ["number", "date", "string"]


def _golden_type(golden: str) -> str:
    v = wtq_eval.to_value(str(golden))
    return {"NumberValue": "number", "DateValue": "date"}.get(type(v).__name__, "string")


def _is_no_output(v: dict) -> bool:
    out = (v.get("output") or "").strip()
    msg = (v.get("message") or "").lower()
    return out == "" or "not found" in msg or "not in output" in msg or "read error" in msg


def _eval_files(eval_dir: Path) -> list[Path]:
    samples = sorted(eval_dir.glob("eval_sample_*.json"))
    if samples:
        return samples
    single = eval_dir / "eval.json"
    return [single] if single.exists() else []


def score_eval_dir(eval_dir: Path) -> dict:
    files = _eval_files(eval_dir)
    if not files:
        raise SystemExit(f"ERROR: no eval.json / eval_sample_*.json in {eval_dir}")

    # Aggregate per-case over samples (a case "passes" if any sample passes —
    # pass@k union; for pass@1 this is just the single sample).
    by_case: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for cid, v in d.items():
            slot = by_case.setdefault(cid, {"pass": False, "any_output": False, "golden": v.get("golden", "")})
            slot["pass"] = slot["pass"] or bool(v.get("pass"))
            slot["any_output"] = slot["any_output"] or not _is_no_output(v)
            if v.get("golden"):
                slot["golden"] = v["golden"]

    n = len(by_case)
    n_pass = sum(1 for c in by_case.values() if c["pass"])
    n_no_output = sum(1 for c in by_case.values() if not c["pass"] and not c["any_output"])
    n_wrong = n - n_pass - n_no_output

    # by answer type
    by_type = {t: {"n": 0, "pass": 0} for t in TYPES}
    for c in by_case.values():
        t = _golden_type(c["golden"])
        by_type[t]["n"] += 1
        by_type[t]["pass"] += int(c["pass"])
    type_acc = {t: (by_type[t]["pass"] / by_type[t]["n"] if by_type[t]["n"] else None) for t in TYPES}

    summary = {
        "eval_dir": str(eval_dir),
        "n_samples": len(files),
        "n_cases": n,
        "denotation_accuracy": (n_pass / n) if n else None,
        "decomposition": {
            "pass": n_pass,
            "wrong_value": n_wrong,
            "no_output": n_no_output,
        },
        "by_answer_type": {t: {"n": by_type[t]["n"], "pass": by_type[t]["pass"],
                               "accuracy": type_acc[t]} for t in TYPES},
    }
    (eval_dir / "wtq_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dirs", nargs="+", required=True,
                   help="One or more WTQ eval dirs (each with eval.json or eval_sample_NN.json)")
    args = p.parse_args()

    rows = [(Path(ed).resolve().name, score_eval_dir(Path(ed).resolve())) for ed in args.eval_dirs]

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "  n/a"

    print("\n" + "═" * 90)
    print("  WikiTableQuestions — denotation accuracy + breakdown")
    print("═" * 90)
    print(f"  {'eval dir':<20s} {'cases':>5s} {'acc':>7s} | {'pass':>5s} {'wrong':>6s} {'no_out':>6s} | "
          f"{'num':>7s} {'date':>7s} {'str':>7s}")
    print(f"  {'-'*20:<20s} {'-'*5:>5s} {'-'*7:>7s} | {'-'*5:>5s} {'-'*6:>6s} {'-'*6:>6s} | "
          f"{'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s}")
    for name, s in rows:
        dc = s["decomposition"]; bt = s["by_answer_type"]
        print(f"  {name:<20s} {s['n_cases']:>5d} {pct(s['denotation_accuracy']):>7s} | "
              f"{dc['pass']:>5d} {dc['wrong_value']:>6d} {dc['no_output']:>6d} | "
              f"{pct(bt['number']['accuracy']):>7s} {pct(bt['date']['accuracy']):>7s} {pct(bt['string']['accuracy']):>7s}")
    print("\n  acc = official WTQ denotation accuracy (== headline pass rate)")
    print("  decomposition: pass / wrong_value (answered but mismatched) / no_output (no answer cell)")
    print("  num/date/str = accuracy on number/date/string gold answers")
    print("  files : <eval-dir>/wtq_summary.json\n")


if __name__ == "__main__":
    main()
