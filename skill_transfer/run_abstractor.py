#!/usr/bin/env python3
"""
Run the TrajectoryAbstractor on a pair of trajectories (strong + weak) and
write human-inspectable ExecutionStructure JSON artifacts to disk.

Example:
    export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" 
    
    python3.11 run_abstractor.py \
        --strong cases/example/gpt5-4_trajectory.md \
        --weak   cases/example/new_gpt5-4-mini_trajectory.md \
        --task   cases/example/INSTRUCTION.md \
        --out    cases/example/new_structures/ \
        --model  gpt-5.4 \
        --strong-id gpt-5.4 \
        --weak-id   gpt-5.4-mini \
        --joint
    
      python3.11 run_abstractor.py \
      --strong-structure cases/example/structures/strong_structure.json \
      --strong           cases/example/gpt5-4_trajectory.md \
      --weak             cases/example/new_gpt5-4-mini_trajectory.md \
      --task             cases/example/INSTRUCTION.md \
      --out              cases/example/structures_after_updating/ \
      --weak-id          gpt5-4-mini-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from openai_setup import setup_openai
setup_openai("abstractor")

# ── colour helpers ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _print_structure_summary(label: str, structure, color: str) -> None:
    print(f"\n{BOLD}{_c(label, color)}{RESET}")
    print(f"  agent_id : {structure.agent_id}")
    print(f"  outcome  : {structure.outcome}")
    print(f"  steps    : {structure.total_steps}")
    print(f"  files    : {structure.produced_files or '(none)'}")
    print(f"  nodes:")
    for node in structure.nodes:
        cw_short = node.completes_when[:60] + "…" if len(node.completes_when) > 60 else node.completes_when
        prod = ", ".join(node.produces) if node.produces else "—"
        print(f"    [{node.category_tier1:8}] {node.category_tier2:35}  produces: {prod}")
        print(f"              completes_when: {cw_short}")


def _print_category_diff(strong, weak) -> None:
    strong_t1 = strong.tier1_categories()
    weak_t1   = weak.tier1_categories()
    only_strong = strong_t1 - weak_t1
    only_weak   = weak_t1 - strong_t1
    shared      = strong_t1 & weak_t1

    print(f"\n{BOLD}── Tier-1 Category Diff ──────────────────────{RESET}")
    print(f"  Shared      : {', '.join(sorted(shared)) or '(none)'}")
    if only_strong:
        print(f"  {_c('Only strong', GREEN)} : {', '.join(sorted(only_strong))}")
    if only_weak:
        print(f"  {_c('Only weak  ', RED)}  : {', '.join(sorted(only_weak))}")

    if only_strong:
        print(f"\n{BOLD}Potential gaps (tier1 categories in strong but absent in weak):{RESET}")
        for cat in sorted(only_strong):
            nodes = strong.nodes_by_tier1(cat)
            for n in nodes:
                print(f"  {_c('MISSING', YELLOW)} [{cat}] {n.category_tier2}")
                print(f"           intent: {n.agent_intent}")
                print(f"           completes_when: {n.completes_when}")

    # Also check: shared tier1 but different completes_when depth
    if shared:
        print(f"\n{BOLD}Shared tier1 nodes — completes_when comparison:{RESET}")
        for cat in sorted(shared):
            s_nodes = strong.nodes_by_tier1(cat)
            w_nodes = weak.nodes_by_tier1(cat)
            for s, w in zip(s_nodes, w_nodes):
                print(f"  [{cat}]")
                print(f"    strong: {s.completes_when}")
                print(f"    weak  : {w.completes_when}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abstract strong and weak agent trajectories into ExecutionStructure JSON."
    )
    parser.add_argument("--strong", default=None, help="Path to strong agent trajectory markdown.")
    parser.add_argument("--weak",   default=None,
                        help="Path to weak agent trajectory markdown.  Not required "
                             "when --strong-only is set.")
    parser.add_argument("--task",   required=True, help="Path to task instruction file (or literal string).")
    parser.add_argument(
        "--strong-only", action="store_true",
        help=(
            "Abstract ONLY the strong trajectory into strong_structure.json and exit. "
            "No weak trajectory needed.  Use this to pre-build a fixed strong "
            "reference that many weak abstractions can later reuse via "
            "--strong-structure (keeps the strong baseline identical across runs)."
        ),
    )
    parser.add_argument(
        "--strong-structure", default=None,
        help=(
            "Path to an existing strong_structure.json.  When provided, the strong "
            "trajectory is NOT re-abstracted; the existing structure is reused.  "
            "--strong (the trajectory) should also be supplied so the abstractor "
            "has both the structure and the raw trajectory as reference context "
            "when abstracting the new weak trajectory.  "
            "Useful for comparing a new weak trace against the original strong baseline."
        ),
    )
    parser.add_argument("--out",    required=True, help="Output directory for JSON artifacts.")
    parser.add_argument("--model",  default="gpt-5.4", help="LLM model to use for abstraction.")
    parser.add_argument("--reasoning-effort", default="",
                        help="Reasoning effort level: 'low', 'medium', or 'high'. Empty = not set.")
    parser.add_argument("--strong-id", default=None, help="Agent ID for strong agent (defaults to filename stem).")
    parser.add_argument("--weak-id",   default=None, help="Agent ID for weak agent (defaults to filename stem).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--no-viz", action="store_true", help="Skip graph visualisation.")
    parser.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"],
                        help="Output format for visualisation (default: svg).")
    parser.add_argument(
        "--cost-file", default="",
        help="If set, write a JSON cost summary for this run to this path.",
    )
    parser.add_argument(
        "--raw-log", default="",
        help="If set, write a raw event log to this path.",
    )
    parser.add_argument(
        "--joint", action="store_true",
        help=(
            "Abstract both trajectories together in a single agent run instead of "
            "two independent runs.  The agent sees both trajectories at once and uses "
            "consistent tier1/tier2 labels across both structures."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    # ── Resolve inputs ──────────────────────────────────────────────────────
    out_dir     = Path(args.out)

    # Task: file or literal string
    task_path = Path(args.task)
    if task_path.exists():
        task = task_path.read_text(encoding="utf-8").strip()
    else:
        task = args.task  # treat as literal string

    # Validate flag combinations.
    if args.strong_only:
        if not args.strong:
            print(f"{_c('ERROR', RED)}: --strong-only requires --strong (the trajectory to abstract).",
                  file=sys.stderr)
            sys.exit(1)
        weak_path = None
    else:
        if not args.weak:
            print(f"{_c('ERROR', RED)}: --weak is required (omit it only with --strong-only).",
                  file=sys.stderr)
            sys.exit(1)
        weak_path = Path(args.weak)
        # Need either --strong or --strong-structure (or both)
        if not args.strong and not args.strong_structure:
            print(
                f"{_c('ERROR', RED)}: provide --strong (trajectory) or "
                f"--strong-structure (existing JSON), or both.",
                file=sys.stderr,
            )
            sys.exit(1)

    if weak_path is not None:
        if not weak_path.exists():
            print(f"{_c('ERROR', RED)}: weak trajectory not found: {weak_path}", file=sys.stderr)
            sys.exit(1)
        weak_md = weak_path.read_text(encoding="utf-8")
    else:
        weak_md = ""  # --strong-only: no weak trajectory

    # ── Resolve strong-side inputs ──────────────────────────────────────────
    strong_structure_json: str | None = None
    reuse_strong = bool(args.strong_structure)

    if reuse_strong:
        strong_struct_path = Path(args.strong_structure)
        if not strong_struct_path.exists():
            print(
                f"{_c('ERROR', RED)}: --strong-structure file not found: {strong_struct_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        strong_structure_json = strong_struct_path.read_text(encoding="utf-8")
        # --strong (raw trajectory) is optional but recommended for richer reference
        if args.strong:
            strong_path = Path(args.strong)
            if not strong_path.exists():
                print(
                    f"{_c('WARNING', YELLOW)}: --strong trajectory not found: {strong_path}"
                    f" — will reference structure only.",
                    file=sys.stderr,
                )
                strong_md = ""
            else:
                strong_md = strong_path.read_text(encoding="utf-8")
            strong_id = args.strong_id or strong_path.stem
        else:
            strong_md = ""
            print(
                f"{_c('WARNING', YELLOW)}: --strong not provided; weak abstractor will "
                f"reference the structure JSON only (no raw trajectory context).",
                file=sys.stderr,
            )
            import json as _json
            strong_id = args.strong_id or _json.loads(strong_structure_json).get("agent_id", "strong")
            strong_md = ""
    else:
        strong_path = Path(args.strong)
        strong_id   = args.strong_id or strong_path.stem
        if not strong_path.exists():
            print(f"{_c('ERROR', RED)}: strong trajectory not found: {strong_path}", file=sys.stderr)
            sys.exit(1)
        strong_md = strong_path.read_text(encoding="utf-8")

    weak_id = args.weak_id or (weak_path.stem if weak_path is not None else "weak")

    # ── Import abstractor (deferred so --help works without dependencies) ───
    from trajectory_abstractor import TrajectoryAbstractor  # noqa: PLC0415

    _model_kwargs: dict | None = None
    if args.reasoning_effort:
        from agents import ModelSettings  # noqa: PLC0415
        _model_kwargs = {"model_settings": ModelSettings(reasoning={"effort": args.reasoning_effort})}

    abstractor = TrajectoryAbstractor(model=args.model, model_kwargs=_model_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    strong_out       = out_dir / "strong_structure.json"
    weak_out         = out_dir / "weak_structure.json"

    # ── --strong-only: abstract strong, write JSON, and exit ─────────────────
    if args.strong_only:
        strong_traj_out = out_dir / "strong_abstractor_trajectory.md"
        print(f"\n{BOLD}Abstracting strong agent trajectory ({strong_id}) — strong-only mode...{RESET}")
        raw_log_path = Path(args.raw_log) if args.raw_log else None
        strong_structure = abstractor.abstract(
            trajectory_md=strong_md,
            agent_id=strong_id,
            task=task,
            output_path=strong_out,
            trajectory_path=strong_traj_out,
            raw_log_path=raw_log_path,
        )
        print(f"  {_c('Saved', GREEN)}: {strong_out}")
        print(f"  {_c('Saved', GREEN)}: {strong_traj_out}")
        _print_structure_summary(f"Strong agent: {strong_id}", strong_structure, GREEN)
        # Cost tracking (same pattern as the main path below)
        from cost_tracker import CostTracker, CostSummary  # noqa: PLC0415
        _summary = CostSummary()
        for i, stream_result in enumerate(abstractor._stream_results):
            _tracker = CostTracker(model=args.model, label=f"abstractor_run_{i+1}")
            _tracker.observe(stream_result)
            if _tracker.run_cost:
                _summary.add(_tracker.run_cost)
        _summary.print_summary(title="Abstractor Cost (strong-only)")
        if args.cost_file:
            _summary.save(args.cost_file)
        print(f"\n{BOLD}Artifacts written to: {out_dir}{RESET}")
        print(f"  {strong_out.name}  (strong only)")
        return

    if reuse_strong:
        # ── Reuse existing strong structure; only abstract weak ─────────────
        from execution_structure import ExecutionStructure as _ES
        strong_structure = _ES.model_validate(
            __import__("json").loads(strong_structure_json)
        )
        # Copy to out_dir so downstream tools find it in the expected location
        if not strong_out.exists() or strong_out.resolve() != Path(args.strong_structure).resolve():
            strong_out.write_text(strong_structure_json, encoding="utf-8")

        weak_traj_out = out_dir / "weak_abstractor_trajectory.md"
        print(
            f"\n{BOLD}Reusing strong structure from {args.strong_structure}{RESET}\n"
            f"  {_c('Loaded', CYAN)}: {strong_id} "
            f"({strong_structure.total_steps} nodes, {strong_structure.outcome})"
        )
        ref_note = "structure + trajectory" if strong_md else "structure only"
        print(f"\n{BOLD}Abstracting weak agent trajectory ({weak_id}) with reference ({ref_note})...{RESET}")
        raw_log_path = Path(args.raw_log) if args.raw_log else None
        weak_structure = abstractor.abstract(
            trajectory_md=weak_md,
            agent_id=weak_id,
            task=task,
            output_path=weak_out,
            trajectory_path=weak_traj_out,
            reference_structure_json=strong_structure_json,
            reference_trajectory_md=strong_md or None,
            raw_log_path=raw_log_path,
        )
        print(f"  {_c('Saved', GREEN)}: {weak_out}")
        print(f"  {_c('Saved', GREEN)}: {weak_traj_out}")

    elif args.joint:
        # ── Joint abstraction (both traces in one prompt) ───────────────────
        joint_traj_out = out_dir / "joint_abstractor_trajectory.md"
        print(f"\n{BOLD}Joint abstraction ({strong_id} + {weak_id})...{RESET}")
        raw_log_path = Path(args.raw_log) if args.raw_log else None
        strong_structure, weak_structure = abstractor.abstract_pair(
            strong_md=strong_md,
            weak_md=weak_md,
            strong_id=strong_id,
            weak_id=weak_id,
            task=task,
            strong_output_path=strong_out,
            weak_output_path=weak_out,
            trajectory_path=joint_traj_out,
            raw_log_path=raw_log_path,
        )
        print(f"  {_c('Saved', GREEN)}: {strong_out}")
        print(f"  {_c('Saved', GREEN)}: {weak_out}")
        print(f"  {_c('Saved', GREEN)}: {joint_traj_out}")

    else:
        # ── Separate abstraction (independent runs) ─────────────────────────
        strong_traj_out = out_dir / "strong_abstractor_trajectory.md"
        weak_traj_out   = out_dir / "weak_abstractor_trajectory.md"

        print(f"\n{BOLD}Abstracting strong agent trajectory ({strong_id})...{RESET}")
        raw_log_path = Path(args.raw_log) if args.raw_log else None
        strong_structure = abstractor.abstract(
            trajectory_md=strong_md,
            agent_id=strong_id,
            task=task,
            output_path=strong_out,
            trajectory_path=strong_traj_out,
            raw_log_path=raw_log_path,
        )
        print(f"  {_c('Saved', GREEN)}: {strong_out}")
        print(f"  {_c('Saved', GREEN)}: {strong_traj_out}")

        print(f"\n{BOLD}Abstracting weak agent trajectory ({weak_id})...{RESET}")
        weak_structure = abstractor.abstract(
            trajectory_md=weak_md,
            agent_id=weak_id,
            task=task,
            output_path=weak_out,
            trajectory_path=weak_traj_out,
            raw_log_path=raw_log_path,
        )
        print(f"  {_c('Saved', GREEN)}: {weak_out}")
        print(f"  {_c('Saved', GREEN)}: {weak_traj_out}")

    # ── Print comparison summary ────────────────────────────────────────────
    _print_structure_summary(f"Strong agent: {strong_id}", strong_structure, GREEN)
    _print_structure_summary(f"Weak agent  : {weak_id}",   weak_structure,   YELLOW)
    _print_category_diff(strong_structure, weak_structure)

    # ── Visualise individual graphs ─────────────────────────────────────────
    if not args.no_viz:
        from execution_visualizer import visualize_structure  # noqa

        print(f"\n{BOLD}Generating visualisations...{RESET}")
        fmt = args.fmt

        rendered_s = visualize_structure(
            strong_structure, out_dir / f"strong_structure.{fmt}",
            title=f"{strong_id} — execution structure",
        )
        print(f"  {_c('Saved', GREEN)}: {rendered_s}")

        rendered_w = visualize_structure(
            weak_structure, out_dir / f"weak_structure.{fmt}",
            title=f"{weak_id} — execution structure",
        )
        print(f"  {_c('Saved', GREEN)}: {rendered_w}")

    print(f"\n{BOLD}Artifacts written to: {out_dir}{RESET}")
    print(f"  {strong_out.name}")
    print(f"  {weak_out.name}")
    if reuse_strong:
        print(f"  {weak_traj_out.name}  (weak only — strong reused)")
    elif args.joint:
        print(f"  {joint_traj_out.name}")
    else:
        print(f"  {strong_traj_out.name}")
        print(f"  {weak_traj_out.name}")
    if not args.no_viz:
        print(f"  strong_structure.{args.fmt}  (graph)")
        print(f"  weak_structure.{args.fmt}    (graph)")
    print(f"\n{_c('Next step:', CYAN)} run_gap_diagnoser.py --structures-dir {out_dir} --task-dir <task-dir> --skill-name <skill-name>")

    # ── Cost tracking ─────────────────────────────────────────────────────────
    from cost_tracker import CostTracker, CostSummary
    summary = CostSummary()
    for i, stream_result in enumerate(abstractor._stream_results):
        label = f"abstractor_run_{i+1}"
        tracker = CostTracker(model=args.model, label=label)
        tracker.observe(stream_result)
        if tracker.run_cost:
            summary.add(tracker.run_cost)
    summary.print_summary(title="Abstractor Cost")
    if args.cost_file:
        summary.save(args.cost_file)

    print()


if __name__ == "__main__":
    main()
