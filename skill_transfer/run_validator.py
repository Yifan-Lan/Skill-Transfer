#!/usr/bin/env python3
"""Command-line runner for the Validator.

Compares the weak agent's produced files against the strong agent's for one
case and writes validation_report.json — a finer-grained signal than the
binary task verdict, consumed by the Diagnoser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai_setup import setup_openai
setup_openai("validator")

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the hybrid ValidatorAgent (programmatic + LLM)."
    )
    parser.add_argument("--strong-structure", required=True,
                        help="Path to strong agent's ExecutionStructure JSON.")
    parser.add_argument("--weak-structure", required=True,
                        help="Path to weak agent's ExecutionStructure JSON.")
    parser.add_argument("--strong-results", default=None,
                        help="Directory containing output files from the strong agent.")
    parser.add_argument("--weak-results", default=None,
                        help="Directory containing output files from the weak agent.")
    parser.add_argument("--task", default=None,
                        help="Path to task instruction file (or literal string).")
    parser.add_argument("--out", required=True,
                        help="Output directory for validation_report.json and validator_feedback.txt.")
    parser.add_argument("--model", default="gpt-5.4",
                        help="LLM model for the validator agent (default: gpt-5.4).")
    parser.add_argument("--strong-id", default=None,
                        help="Override strong agent ID (defaults to structure's agent_id).")
    parser.add_argument("--weak-id", default=None,
                        help="Override weak agent ID (defaults to structure's agent_id).")
    parser.add_argument(
        "--cost-file", default="",
        help="If set, write a JSON cost summary for this run to this path.",
    )
    parser.add_argument(
        "--trajectory-file", default="",
        help="If set, write a markdown trajectory log to this path (default: <out>/validator_trajectory.md).",
    )
    parser.add_argument(
        "--raw-log-file", default="",
        help="If set, write raw event log to this path.",
    )
    parser.add_argument(
        "--golden-answer", default="",
        help="Path to golden_answer.txt for text-answer tasks (e.g. OfficeQA). "
             "When provided, a golden-comparator tool is used instead of compare_files "
             "(which one depends on --golden-kind).",
    )
    parser.add_argument(
        "--golden-kind", default="officeqa", choices=["officeqa", "dabench"],
        help="Format of --golden-answer: 'officeqa' (single scalar, tolerance-fuzzy-matched "
             "via compare_answer_fuzzy) or 'dabench' (JSON [[name,value],...] pairs against a "
             "multi-value @name[value] output, matched via compare_answer_dabench: per "
             "sub-answer exact-string-or-1e-6 match, ALL must match to pass).",
    )
    parser.add_argument(
        "--reasoning-effort", default="",
        help="Reasoning effort level: 'low', 'medium', or 'high'. Empty = not set.",
    )
    args = parser.parse_args()

    # ── Load structures ────────────────────────────────────────────────────────
    from execution_structure import ExecutionStructure

    strong_struct_path = Path(args.strong_structure)
    weak_struct_path   = Path(args.weak_structure)

    for p, label in [(strong_struct_path, "strong-structure"), (weak_struct_path, "weak-structure")]:
        if not p.exists():
            print(f"{_c('ERROR', RED)}: --{label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    strong_structure = ExecutionStructure.model_validate(
        json.loads(strong_struct_path.read_text(encoding="utf-8"))
    )
    weak_structure = ExecutionStructure.model_validate(
        json.loads(weak_struct_path.read_text(encoding="utf-8"))
    )

    strong_id = args.strong_id or strong_structure.agent_id
    weak_id   = args.weak_id   or weak_structure.agent_id

    # ── Resolve directories ────────────────────────────────────────────────────
    strong_results_dir = Path(args.strong_results) if args.strong_results else None
    weak_results_dir   = Path(args.weak_results)   if args.weak_results   else None
    out_dir            = Path(args.out)

    if strong_results_dir and not strong_results_dir.is_dir():
        print(f"{_c('ERROR', RED)}: --strong-results is not a directory: {strong_results_dir}", file=sys.stderr)
        sys.exit(1)
    if weak_results_dir and not weak_results_dir.is_dir():
        print(f"{_c('ERROR', RED)}: --weak-results is not a directory: {weak_results_dir}", file=sys.stderr)
        sys.exit(1)

    # Symmetry check: both or neither — but only when there is no golden answer.
    # OfficeQA passes --weak-results alone (no --strong-results) alongside
    # --golden-answer, which is intentional; don't null them out in that case.
    if (strong_results_dir is None) != (weak_results_dir is None) and not args.golden_answer:
        print(
            f"{_c('WARNING', YELLOW)}: provide both --strong-results and --weak-results "
            "for file comparison; skipping file comparison.",
            file=sys.stderr,
        )
        strong_results_dir = weak_results_dir = None

    # ── Task description ───────────────────────────────────────────────────────
    task = ""
    if args.task:
        task_path = Path(args.task)
        task = task_path.read_text(encoding="utf-8").strip() if task_path.exists() else args.task

    golden_answer_file = Path(args.golden_answer) if args.golden_answer else None
    if golden_answer_file and not golden_answer_file.exists():
        print(f"{_c('ERROR', RED)}: --golden-answer not found: {golden_answer_file}", file=sys.stderr)
        sys.exit(1)

    # ── Run agent or fall back to programmatic-only ────────────────────────────
    if (strong_results_dir and weak_results_dir) or golden_answer_file:
        from validator_agent import ValidatorAgent
        from cost_tracker import CostTracker, CostSummary
        from runner_common import StreamPrinter

        class ValidatorStreamPrinter(StreamPrinter):
            def handle(self, event):
                from runner_common import _get_event_type, GREEN, BOLD, RESET
                e_type = _get_event_type(event)
                if "run_complete" in e_type or "response.done" in e_type:
                    self.flush_thought()
                    self.flush_tool()
                    # Skip the agent's completion message; run_validator prints its own
                    return
                super().handle(event)

        # Resolve trajectory / raw-log paths
        traj_path    = Path(args.trajectory_file) if args.trajectory_file else out_dir / "validator_trajectory.md"
        raw_log_path = Path(args.raw_log_file)    if args.raw_log_file    else None
        traj_path.parent.mkdir(parents=True, exist_ok=True)

        traj_f   = open(traj_path, "w", encoding="utf-8")
        raw_log_f = open(raw_log_path, "w", encoding="utf-8") if raw_log_path else None

        printer = ValidatorStreamPrinter(traj_file=traj_f, raw_log_file=raw_log_f)

        print(f"\n{BOLD}Running ValidatorAgent ({args.model})…{RESET}")
        print(f"{'─' * 60}")

        _model_kwargs: dict | None = None
        if args.reasoning_effort:
            from agents import ModelSettings  # noqa: PLC0415
            _model_kwargs = {"model_settings": ModelSettings(reasoning={"effort": args.reasoning_effort})}
        agent = ValidatorAgent(model=args.model, model_kwargs=_model_kwargs)
        report = agent.validate(
            task=task,
            strong_dir=strong_results_dir or out_dir,
            weak_dir=weak_results_dir or out_dir,
            strong_agent_id=strong_id,
            weak_agent_id=weak_id,
            strong_produced_files=strong_structure.produced_files or [],
            weak_produced_files=weak_structure.produced_files or None,
            golden_answer_file=golden_answer_file,
            golden_answer_kind=args.golden_kind,
            out_dir=out_dir,
            on_event=printer.handle,
        )
        printer.flush_thought()

        traj_f.close()
        if raw_log_f:
            raw_log_f.close()
        print(f"\n{BOLD}{GREEN}[✓] Validation complete!{RESET}")
        print("─" * 60)

        # Cost tracking
        summary = CostSummary()
        for i, sr in enumerate(agent._stream_results):
            tracker = CostTracker(model=args.model, label=f"validator_agent_{i+1}")
            tracker.observe(sr)
            if tracker.run_cost:
                summary.add(tracker.run_cost)
        summary.print_summary(title="Validator Cost")
        if args.cost_file:
            summary.save(args.cost_file)

    else:
        # No result dirs — fall back to structure-only programmatic validation
        from validator import Validator

        print(f"\n{BOLD}Running programmatic validator (no result dirs provided)…{RESET}")
        validator = Validator()
        prog_report = validator.validate(
            strong_structure,
            weak_structure,
            output_path=out_dir / "validation_report.json",
        )
        report = json.loads((out_dir / "validation_report.json").read_text())

        # Write minimal feedback for GapAgent
        feedback_path = out_dir / "validator_feedback.txt"
        rq = prog_report.result_quality
        lines = [
            f"Verdict      : {prog_report.verdict}",
            f"File presence: {rq.file_presence_ratio}",
            f"Content match: {rq.content_match_ratio}",
        ]
        if rq.missing_files:
            lines.append(f"Missing output files (weak did not produce): {rq.missing_files}")
        if rq.extra_files:
            lines.append(f"Extra output files (weak over-produced): {rq.extra_files}")
        feedback_path.write_text("\n".join(lines), encoding="utf-8")

    # ── Print summary ──────────────────────────────────────────────────────────
    verdict  = report.get("verdict", "UNKNOWN")
    v_color  = GREEN if verdict == "PASS" else (YELLOW if verdict == "PARTIAL" else RED)
    rq       = report.get("result_quality", {})
    scores   = rq.get("scores", {})
    files    = rq.get("files", {})

    print(f"\n{BOLD}{_c('═' * 60, CYAN)}{RESET}")
    print(f"  {BOLD}Validation Report{RESET}")
    print(f"  Strong : {strong_id}")
    print(f"  Weak   : {weak_id}")
    print(f"  Verdict: {BOLD}{_c(verdict, v_color)}{RESET}")
    print(f"{BOLD}{_c('═' * 60, CYAN)}{RESET}")

    def _safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    score_color = GREEN if _safe_float(scores.get("overall_quality_score", 0)) >= 0.8 else YELLOW
    print(f"\n{BOLD}── Scores ───────────────────────────────────────────{RESET}")
    
    fpr_raw = scores.get('file_presence_ratio')
    if fpr_raw is None or isinstance(fpr_raw, str):
        print(f"  {YELLOW}[!] Warning: 'scores' not fully resolved or contains invalid types (found: {fpr_raw}). Defaulting values to 0.0.{RESET}")

    fpr = _safe_float(fpr_raw)
    cmr = _safe_float(scores.get('content_match_ratio'))
    print(f"  File presence  : {fpr:.3f}")
    print(f"  Content match  : {cmr:.3f}")
    qs = _safe_float(scores.get("overall_quality_score"))
    print(f"  Quality score  : {_c(f'{qs:.3f}', score_color)}")

    missing = files.get("missing_from_weak", [])
    extra   = files.get("extra_in_weak", [])
    present = files.get("present", [])

    if missing:
        print(f"\n  {_c('Missing files:', RED)}")
        for f in missing:
            print(f"    ✗ {f}")

    if present:
        print(f"\n  File comparisons:")
        for fc in present:
            status = _c("✓", GREEN) if fc.get("content_match") else _c("✗", RED)
            name = fc.get("filename", "?")
            wname = fc.get("matched_weak_filename", "")
            if wname and wname != name:
                name += f"  {_c('→', YELLOW)} {wname}"
            ctype = f"  [{fc.get('comparator', '?')}]"
            d = fc.get("detail", {})
            comp = fc.get("comparator", "")
            detail = ""
            if comp == "tabular":
                rmr = d.get('row_match_ratio')
                rmr_s = f"{rmr:.2f}" if isinstance(rmr, (int, float)) else "?"
                detail = (
                    f"  rows={rmr_s}"
                    f"  cols={'✓' if d.get('columns_match') else '✗'}"
                    f"  missing={d.get('missing_rows', 0)}"
                    f"  extra={d.get('extra_rows', 0)}"
                )
            elif comp == "json" and not fc.get("content_match"):
                os_ = d.get("only_in_strong", [])
                ow  = d.get("only_in_weak", [])
                if os_:
                    detail += f"  only_strong={os_}"
                if ow:
                    detail += f"  only_weak={ow}"
            elif comp == "text":
                detail = f"  lines: {d.get('strong_lines','?')} → {d.get('weak_lines','?')}"
            print(f"    {status} {name}{ctype}{detail}")

    if extra:
        print(f"\n  {_c('Extra files in weak:', YELLOW)}")
        for f in extra:
            print(f"    + {f}")

    if report.get("analysis"):
        print(f"\n{BOLD}── Analysis ─────────────────────────────────────────{RESET}")
        print(f"  {report['analysis'][:300]}")

    report_path   = out_dir / "validation_report.json"
    feedback_path = out_dir / "validator_feedback.txt"
    print(f"\n{_c('Saved:', GREEN)} {report_path}")
    print(f"{_c('Saved:', GREEN)} {feedback_path}")
    print()


if __name__ == "__main__":
    main()
