#!/usr/bin/env python3
"""Fine-grained InfiAgent-DABench breakdown from existing eval JSONs.

Mirrors DABench's own official fine-grained metrics from `eval_closed_form.py`
(see the DABench repository) — reused, not reinvented:

  1. Accuracy by Question        — headline pass rate (ALL sub-answers correct).
                                    Computed over ALL prepared cases (a missing
                                    output.txt counts as wrong) — same denominator
                                    as our own pass@1, unlike the official script
                                    which only counts cases with a found response.
  2. Accuracy by Sub-Question    — micro-average: sum(correct sub-answers) /
                                    sum(total sub-answers), over cases that
                                    produced a parseable answer (a case with NO
                                    output.txt has no sub-answers to score and is
                                    excluded from this denominator — matching the
                                    official script's behavior for missing responses).
  3. Accuracy Proportional by Sub-Question — macro-average of each case's own
                                    partial-credit score (= our `sub_score`
                                    field), i.e. every question weighted equally
                                    regardless of how many sub-answers it has.
                                    Same case-inclusion rule as #2.
  4. Concept Accuracy            — per DABench `concepts` tag (Summary
                                    Statistics, Outlier Detection, ...), fraction
                                    of ALL-CORRECT questions mentioning that tag.
                                    Same denominator as #1 (all cases).
  5. Concept-Count Accuracy      — accuracy grouped by how many concepts tag a
                                    question (1, 2, 3, ...), plus the official
                                    ">= 2 concepts" slice. Same denominator as #1.

All from the per-case {pass, correctness, sub_score} already in eval.json (our
`_eval_dabench` return shape) + `concepts` from the source dataset.json's per-
task record — no agent re-run, no API calls.

Output (per eval dir): `dabench_summary.json`, plus a comparison table.

Usage:
    python compute_dabench_breakdown.py --eval-dirs <dir1> <dir2> ... \
        --dataset-dir data/dabench/test_150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _eval_files(eval_dir: Path) -> list[Path]:
    samples = sorted(eval_dir.glob("eval_sample_*.json"))
    if samples:
        return samples
    single = eval_dir / "eval.json"
    return [single] if single.exists() else []


def _load_concepts(dataset_dir: Path) -> dict[str, list[str]]:
    dj = dataset_dir / "dataset.json"
    if not dj.exists():
        return {}
    return {str(t["id"]): t.get("concepts", []) for t in json.loads(dj.read_text(encoding="utf-8"))}


def score_eval_dir(eval_dir: Path, concepts_by_id: dict[str, list[str]]) -> dict:
    files = _eval_files(eval_dir)
    if not files:
        raise SystemExit(f"ERROR: no eval.json / eval_sample_*.json in {eval_dir}")

    # Aggregate per-case over samples: pass@k union for `pass`; a sub-answer is
    # "correct" if correct in ANY sample (consistent union semantics).
    by_case: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for cid, v in d.items():
            slot = by_case.setdefault(cid, {"pass": False, "correctness": None})
            slot["pass"] = slot["pass"] or bool(v.get("pass"))
            corr = v.get("correctness")
            if corr:
                if slot["correctness"] is None:
                    slot["correctness"] = dict(corr)
                else:
                    for k, ok in corr.items():
                        slot["correctness"][k] = slot["correctness"].get(k, False) or bool(ok)

    n = len(by_case)
    n_pass = sum(1 for c in by_case.values() if c["pass"])
    accuracy_by_question = (n_pass / n) if n else None

    # sub-question-level: only cases with a parseable answer (correctness present)
    scored = [c for c in by_case.values() if c["correctness"]]
    sub_correct = sum(sum(1 for ok in c["correctness"].values() if ok) for c in scored)
    sub_total = sum(len(c["correctness"]) for c in scored)
    accuracy_by_sub_question = (sub_correct / sub_total) if sub_total else None

    per_case_sub_scores = [
        sum(1 for ok in c["correctness"].values() if ok) / len(c["correctness"])
        for c in scored
    ]
    accuracy_proportional = (sum(per_case_sub_scores) / len(scored)) if scored else None

    # concept breakdown (over ALL cases, missing-output counts as wrong)
    concept_acc: dict[str, dict] = {}
    count_acc: dict[int, dict] = {}
    two_plus = {"n": 0, "pass": 0}
    for cid, c in by_case.items():
        concepts = concepts_by_id.get(cid, [])
        ok = c["pass"]
        for concept in concepts:
            slot = concept_acc.setdefault(concept, {"n": 0, "pass": 0})
            slot["n"] += 1
            slot["pass"] += int(ok)
        cnt = len(concepts)
        slot = count_acc.setdefault(cnt, {"n": 0, "pass": 0})
        slot["n"] += 1
        slot["pass"] += int(ok)
        if cnt >= 2:
            two_plus["n"] += 1
            two_plus["pass"] += int(ok)

    concept_accuracy = {k: (v["pass"] / v["n"] if v["n"] else None) for k, v in concept_acc.items()}
    concept_count_accuracy = {str(k): (v["pass"] / v["n"] if v["n"] else None) for k, v in count_acc.items()}
    two_plus_accuracy = (two_plus["pass"] / two_plus["n"]) if two_plus["n"] else None

    summary = {
        "eval_dir": str(eval_dir),
        "n_samples": len(files),
        "n_cases": n,
        "n_cases_with_answer": len(scored),
        "accuracy_by_question": accuracy_by_question,
        "accuracy_by_sub_question": accuracy_by_sub_question,
        "accuracy_proportional_by_sub_question": accuracy_proportional,
        "concept_accuracy": {k: {"n": concept_acc[k]["n"], "pass": concept_acc[k]["pass"], "accuracy": v}
                             for k, v in concept_accuracy.items()},
        "concept_count_accuracy": {k: {"n": count_acc[int(k)]["n"], "pass": count_acc[int(k)]["pass"], "accuracy": v}
                                   for k, v in concept_count_accuracy.items()},
        "accuracy_two_or_more_concepts": two_plus_accuracy,
    }
    (eval_dir / "dabench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dirs", nargs="+", required=True,
                   help="One or more DABench eval dirs (each with eval.json or eval_sample_NN.json)")
    p.add_argument("--dataset-dir", required=True,
                   help="DABench dataset dir (for dataset.json's per-task `concepts` field)")
    args = p.parse_args()

    concepts_by_id = _load_concepts(Path(args.dataset_dir).resolve())
    rows = [(Path(ed).resolve().name, score_eval_dir(Path(ed).resolve(), concepts_by_id))
            for ed in args.eval_dirs]

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "  n/a"

    print("\n" + "═" * 100)
    print("  InfiAgent-DABench — accuracy by question / sub-question + concept breakdown")
    print("═" * 100)
    print(f"  {'eval dir':<24s} {'cases':>5s} {'by-Q':>7s} | {'by-subQ':>8s} {'prop-subQ':>10s} | {'>=2 concepts':>12s}")
    print(f"  {'-'*24:<24s} {'-'*5:>5s} {'-'*7:>7s} | {'-'*8:>8s} {'-'*10:>10s} | {'-'*12:>12s}")
    for name, s in rows:
        print(f"  {name:<24s} {s['n_cases']:>5d} {pct(s['accuracy_by_question']):>7s} | "
              f"{pct(s['accuracy_by_sub_question']):>8s} {pct(s['accuracy_proportional_by_sub_question']):>10s} | "
              f"{pct(s['accuracy_two_or_more_concepts']):>12s}")

    print("\n  Concept accuracy (per eval dir):")
    all_concepts = sorted({k for _, s in rows for k in s["concept_accuracy"]})
    for concept in all_concepts:
        line = f"    {concept:<32s}"
        for _, s in rows:
            c = s["concept_accuracy"].get(concept)
            line += f"  {pct(c['accuracy']) if c else '  n/a':>7s}({c['n'] if c else 0})"
        print(line)

    print("\n  by-Q      = accuracy by question (== headline pass rate; ALL cases, missing output = wrong)")
    print("  by-subQ   = accuracy by sub-question (micro-avg over cases with a parseable answer)")
    print("  prop-subQ = accuracy proportional by sub-question (macro-avg of per-case sub_score)")
    print("  files     : <eval-dir>/dabench_summary.json\n")


if __name__ == "__main__":
    main()
