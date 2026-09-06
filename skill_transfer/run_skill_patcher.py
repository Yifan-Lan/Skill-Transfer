#!/usr/bin/env python3
"""Command-line runner for SkillPatcher.

Reads the gap report written by run_gap_diagnoser.py and applies every patch
hint to the skill in one pass.  Retries on API errors and on text-only early
stops; on terminal failure the skill files are restored from a snapshot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from openai_setup import setup_openai
setup_openai("skill-patcher")

from runner_common import (
    RESET, BOLD, GREEN, BLUE, YELLOW, CYAN, RED, DIM,
    _c,
    _format_retry_exception,
    _snapshot_files,
    _restore_file_snapshots,
    _archive_retry_artifacts,
    _is_stale_session_error,
    StreamPrinter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repository root


# ──────────────────────────────────────────────────────────────────────────────
# extract_patcher_narrative — Patcher-specific narrative extractor
#
# Unlike GapAgent's extract_patch_narrative() (which filters at the
# submit_gap_report boundary), the Patcher's whole trajectory is patch
# content.  We just compress observations from read_file / fallback
# tools so the resulting JSON is small enough to feed back into next
# iter's Diagnoser via --prev-gap-patch-narrative-files.
# ──────────────────────────────────────────────────────────────────────────────

# Tools whose ARGUMENTS we keep verbatim but whose observations we drop
_KEEP_ARGS_DROP_OBS = {
    "replace_in_file",
    "apply_patch",
    "read_prev_case_structures",
}
# Fallback tools — keep args, drop observations
_FALLBACK_TOOLS = {
    "read_case_structures",
    "read_prev_case_structures",
    "read_case_validator_report",
    "read_case_trajectory",
}


def extract_patcher_narrative(traj_json_path: Path) -> Path | None:
    """Compact narrative of a Patcher session.

    Keeps:
      • thought steps (assistant reasoning, verbatim)
      • replace_in_file / apply_patch call arguments (verbatim, no obs)
      • Fallback tool call arguments (no obs)
      • activate_skill call (records skill activation event)
      • finalize_patches call (completion marker)

    Drops:
      • read_file observations (bulky — skill content)
      • shell observations (bulky — script outputs)
      • generate_viz observations (binary-ish)

    Saves to <traj_json_path.parent>/gap_patch_narrative.json — same
    filename as GapAgent's narrative so downstream tooling
    (Diagnoser's --prev-gap-patch-narrative-files) doesn't need to
    change.
    """
    try:
        data = json.loads(traj_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    steps = data.get("steps", [])
    kept: list[dict] = []
    for s in steps:
        if s.get("type") == "thought":
            kept.append({"type": "thought", "text": s.get("text", "")})
            continue
        if s.get("type") != "tool_call":
            continue
        tool = s.get("tool", "")
        args = s.get("args", {}) or {}
        if tool in _KEEP_ARGS_DROP_OBS or tool in _FALLBACK_TOOLS:
            kept.append({"type": "tool_call", "tool": tool, "args": args})
        elif tool == "activate_skill":
            kept.append({"type": "tool_call", "tool": tool, "args": args})
        elif tool == "finalize_patches":
            kept.append({"type": "tool_call", "tool": tool, "args": args})
        # Else (read_file, write_file, shell, generate_viz) — dropped.

    out_path = traj_json_path.parent / "gap_patch_narrative.json"
    out_path.write_text(
        json.dumps(
            {"format": "patcher_narrative_v1", "steps": kept},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path


async def main():
    parser = argparse.ArgumentParser(
        description="Run SkillPatcher: apply the patches described in a "
                    "GapDiagnoser-produced gap_report.json.",
    )
    # ── primary input ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--gap-report-path", required=True,
        help="Path to the gap_report.json produced by GapDiagnoser.  This is "
             "the binding input — each gap's skill_patch_hint is the patch blueprint.",
    )
    # ── case context (fallback only) ──────────────────────────────────────────
    parser.add_argument(
        "--cases-dir", required=True,
        help="Per-batch weak cases dir.  Used by FALLBACK case-reader tools "
             "(read_case_structures etc.) — not for primary iteration.",
    )
    parser.add_argument(
        "--global-cases-dir", default="",
        help="Per-batch strong cases dir.  Defaults to --cases-dir.",
    )
    parser.add_argument(
        "--iter", type=int, default=0,
        help="Iteration index for the cases dir layout (default: 0).",
    )
    # ── skills ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--skill-dir", default=str(PROJECT_ROOT / "skills"),
        help="Root directory of the working skills library.",
    )
    parser.add_argument(
        "--skill-names", nargs="+", required=True,
        help="One or more working skill names (e.g. 'xlsx' or 'pdf xlsx'). "
             "Must include every `target_skill` referenced in gap_report.json.",
    )
    parser.add_argument(
        "--meta-skills-dir", default=str(PROJECT_ROOT / "skills" / "meta_skill_creator"),
        help="Directory of meta-skills (default: skill-creator).",
    )
    # ── previous-iter evidence (fallback only) ────────────────────────────────
    parser.add_argument(
        "--prev-gap-patch-narrative-files", nargs="+", default=[],
        help="Fallback context only.  Diagnoser's skill_patch_hint should "
             "already incorporate this evidence in PRIOR-PATCH CRITIQUE blocks.",
    )
    # ── model / output ────────────────────────────────────────────────────────
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--log-dir", default="",
        help="Directory for trajectory, sentinel, session DB, and narrative output.",
    )
    parser.add_argument("--no-raw-log", action="store_true")
    parser.add_argument("--cost-file", default="")
    parser.add_argument(
        "--trajectory-format", default="json", choices=["md", "json"],
    )
    parser.add_argument(
        "--instruction-save-path", default="",
        help="If set, save the full prompt sent to the patcher to this file.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=50,
    )
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--min-gap-count", type=int, default=1,
                        help="Mirrors Diagnoser flag — Patcher just plumbs through "
                             "for consistency (Patcher itself doesn't filter gaps).")
    args = parser.parse_args()

    skill_dir    = Path(args.skill_dir).resolve()
    project_root = Path(args.project_root).resolve()

    _model_kwargs: dict | None = None
    if args.reasoning_effort or args.max_output_tokens:
        from agents import ModelSettings  # noqa: PLC0415
        _model_kwargs = {"model_settings": ModelSettings(
            reasoning={"effort": args.reasoning_effort} if args.reasoning_effort else None,
            max_tokens=args.max_output_tokens if args.max_output_tokens else None,
        )}

    skill_files = [skill_dir / name / "SKILL.md" for name in args.skill_names]
    for sf in skill_files:
        if not sf.exists():
            print(f"{RED}[!] SKILL.md not found: {sf}{RESET}", file=sys.stderr)
            sys.exit(1)

    gap_report_path = Path(args.gap_report_path).resolve()
    if not gap_report_path.exists():
        print(f"{RED}[!] gap_report.json not found: {gap_report_path}{RESET}",
              file=sys.stderr)
        sys.exit(1)

    # Load + validate gap_report structure for early failure detection.
    try:
        gap_report = json.loads(gap_report_path.read_text(encoding="utf-8"))
        if "gaps" not in gap_report or "recommended_patch_order" not in gap_report:
            print(f"{RED}[!] gap_report.json missing required keys 'gaps' or "
                  f"'recommended_patch_order'{RESET}", file=sys.stderr)
            sys.exit(1)
        n_gaps = len(gap_report.get("gaps", []))
        n_patches = len(gap_report.get("recommended_patch_order", []))
    except Exception as exc:
        print(f"{RED}[!] Failed to parse gap_report.json: {exc}{RESET}",
              file=sys.stderr)
        sys.exit(1)

    # ── build case contexts for fallback tools ────────────────────────────────
    cases_root = Path(args.cases_dir).resolve()
    global_cases_root = (
        Path(args.global_cases_dir).resolve()
        if args.global_cases_dir else cases_root
    )
    iter_n = args.iter
    log_dir = Path(args.log_dir).resolve() if args.log_dir else cases_root
    log_dir.mkdir(parents=True, exist_ok=True)

    from skill_patcher import SkillPatcher
    from gap_base import CaseContext

    case_contexts = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        gcd = global_cases_root / case_dir.name
        s_struct = gcd / "strong_structure.json"
        w_struct = case_dir / f"iter_{iter_n:02d}" / "weak_structure.json"
        _sj = gcd / "strong_trajectory.json"
        s_traj = _sj if _sj.exists() else gcd / "strong_trajectory.md"
        _wj = case_dir / f"iter_{iter_n:02d}" / "weak_trajectory.json"
        w_traj = _wj if _wj.exists() else case_dir / f"iter_{iter_n:02d}" / "weak_trajectory.md"
        if s_struct.exists() and w_struct.exists():
            case_contexts.append(CaseContext(
                case_id=case_dir.name,
                strong_structure_path=str(s_struct),
                weak_structure_path=str(w_struct),
                strong_traj_path=str(s_traj),
                weak_traj_path=str(w_traj),
            ))

    print(f"\n{BOLD}Skill Patcher{RESET}")
    print(f"  model              : {args.model}")
    print(f"  gap_report         : {gap_report_path}")
    print(f"    └─ gaps={n_gaps}, recommended_patch_order={n_patches}")
    print(f"  cases (fallback)   : {len(case_contexts)} from {cases_root}")
    print(f"  log dir            : {log_dir}")
    print(f"  meta-skill         : "
          f"{args.meta_skills_dir}")
    for sf in skill_files:
        print(f"  working skill      : {sf}")

    # ── SQLiteSession with distinct session_id ────────────────────────────────
    from agents.memory.sqlite_session import SQLiteSession
    _session_db_path = log_dir / "patcher_session.sqlite"
    _cross_process_resume = (
        _session_db_path.exists() and _session_db_path.stat().st_size > 512
    )
    if _cross_process_resume:
        print(f"{YELLOW}[session] Existing patcher session DB found — will resume "
              f"({_session_db_path.stat().st_size // 1024} KB).{RESET}")
    _session = SQLiteSession(session_id="skill-patcher-main",
                             db_path=str(_session_db_path))

    def _open_logs(resume: bool = False):
        import datetime as _dt
        mode = "a" if resume else "w"
        raw = (
            None if args.no_raw_log
            else (log_dir / "skill_patcher_raw_log.txt").open(mode, encoding="utf-8")
        )
        traj = (log_dir / "skill_patcher_trajectory.md").open(mode, encoding="utf-8")
        if resume:
            traj.write(f"\n\n---\n**[RESUMED]** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            traj.write(
                f"# Skill Patcher Trajectory\n\n"
                f"**model**: {args.model}  \n"
                f"**gap_report**: {gap_report_path}  \n"
                f"**meta-skill**: {args.meta_skills_dir}\n\n"
            )
        return raw, traj

    raw_log_f, traj_f = _open_logs(resume=_cross_process_resume)

    # ── build the agent ───────────────────────────────────────────────────────
    patcher = SkillPatcher(
        model=args.model,
        project_root=project_root,
        task_dir=cases_root,
        diff_log_path=log_dir / "skill_edits.diff",
        trajectory_format=args.trajectory_format,
        model_kwargs=_model_kwargs,
        min_gap_count=args.min_gap_count,
        meta_skills_dir=args.meta_skills_dir,
        patcher_mode="one-shot",
    )

    # Save the assembled system prompt to the log dir for audit / debugging.
    # Overwritten on every run (including resumes) so the on-disk copy always
    # matches what the live agent is using.
    try:
        (log_dir / "skill_patcher_system_prompt.md").write_text(
            patcher.system_prompt_multi, encoding="utf-8"
        )
    except Exception as exc:
        print(f"{YELLOW}[!] Failed to save system prompt: {exc}{RESET}", file=sys.stderr)

    printer = StreamPrinter(
        traj_file=traj_f,
        raw_log_file=raw_log_f,
        traj_json_path=log_dir / "skill_patcher_trajectory.json",
        preload_steps=_cross_process_resume,
    )

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: (
        print(f"\n\n{RED}[!] Interrupted.{RESET}"), loop.stop()
    ))

    print(f"\n{BOLD}Starting patching…{RESET}\n{'─' * 60}")

    prev_gap_patch_narrative_paths = [
        str(Path(p).resolve())
        for p in args.prev_gap_patch_narrative_files
        if Path(p).exists()
    ]

    instruction_save_path: str | None = None
    if args.instruction_save_path:
        p = Path(args.instruction_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        instruction_save_path = str(p.resolve())

    # ── retry loop ────────────────────────────────────────────────────────────
    _MAX_ERROR_RETRIES = 2
    _MAX_TEXT_ONLY_RETRIES = 8
    _RETRY_DELAY = 10
    _TEXT_ONLY_DELAY = 3
    _error_retries = 0
    _text_only_retries = 0
    _resume_input_override: str | None = None
    _streams: list = []
    _run_ok = False

    # Skill snapshots for terminal-failure restore (Patcher DOES modify skills).
    _skill_snapshots = _snapshot_files(skill_files)

    patches_complete_path = log_dir / "gap_patches_complete"

    try:
        for _attempt in range(_MAX_ERROR_RETRIES + _MAX_TEXT_ONLY_RETRIES + 1):
            _resuming = _attempt > 0 or _cross_process_resume
            try:
                stream = patcher.run_streamed_patcher(
                    task_description=(
                        f"Apply the patches described in {gap_report_path} "
                        f"({n_patches} gaps in recommended_patch_order)."
                    ),
                    cases=case_contexts,
                    skill_paths=[str(sf) for sf in skill_files],
                    out_dir=str(log_dir),
                    gap_report_path=str(gap_report_path),
                    force_read_trajectories=False,
                    prev_gap_trajectory_paths=None,
                    prev_gap_patch_narrative_paths=prev_gap_patch_narrative_paths or None,
                    instruction_save_path=instruction_save_path,
                    case_history_path=None,
                    max_turns=args.max_turns,
                    session=_session,
                    resuming=_resuming,
                    resume_input=_resume_input_override,
                )
                _streams.append(stream)

                _stream_incomplete = False
                async for event in stream.stream_events():
                    printer.handle(event, agent_instance=patcher)
                    _raw_data = getattr(event, "data", None)
                    if _raw_data is not None:
                        _tn = type(_raw_data).__name__
                        if "ResponseIncompleteEvent" in _tn:
                            _stream_incomplete = True
                        elif "ResponseCompletedEvent" in _tn:
                            _stream_incomplete = False

                if _stream_incomplete:
                    raise RuntimeError(
                        "API response truncated (response.incomplete: max_output_tokens exceeded)"
                    )

                # Detect text-only termination: stream ended cleanly but
                # finalize_patches() was never called.  The Patcher's
                # completion sentinel is `gap_patches_complete` (set by
                # finalize_patches tool).
                _text_only_stop = (
                    not patches_complete_path.exists()
                    and _text_only_retries < _MAX_TEXT_ONLY_RETRIES
                )
                if _text_only_stop:
                    _text_only_retries += 1
                    _wait = _TEXT_ONLY_DELAY
                    _resume_input_override = patcher._TEXT_ONLY_RESUME_INPUT
                    print(
                        f"\n{YELLOW}[!] Patcher stopped early (text-only termination) "
                        f"— text-only retry {_text_only_retries}/{_MAX_TEXT_ONLY_RETRIES} "
                        f"in {_wait}s (session preserved)…{RESET}"
                    )
                    traj_f.write(
                        f"\n\n---\n**[TEXT-ONLY RETRY {_text_only_retries}]** "
                        f"resuming after {_wait}s (SQLiteSession carries prior work).\n\n"
                    )
                    traj_f.flush()
                    if raw_log_f:
                        raw_log_f.write(
                            f"\n[retry_text_only] text_only_attempt={_text_only_retries} "
                            f"max={_MAX_TEXT_ONLY_RETRIES} wait_seconds={_wait} "
                            f"session_resume=True\n"
                        )
                        raw_log_f.flush()
                    printer.flush_thought(agent_instance=patcher)
                    printer.flush_tool()
                    printer.finalize_json()
                    if raw_log_f:
                        raw_log_f.close()
                    traj_f.close()
                    raw_log_f, traj_f = _open_logs(resume=True)
                    printer = StreamPrinter(
                        traj_file=traj_f, raw_log_file=raw_log_f,
                        traj_json_path=log_dir / "skill_patcher_trajectory.json",
                        preload_steps=True,
                    )
                    await asyncio.sleep(_wait)
                    continue

                break  # success — exit retry loop

            except Exception as _exc:
                import httpx
                _is_retryable = isinstance(_exc, (httpx.ReadTimeout, httpx.ConnectTimeout,
                                                  httpx.PoolTimeout, TimeoutError))
                if not _is_retryable:
                    try:
                        import httpcore
                        _is_retryable = isinstance(_exc, (httpcore.ReadTimeout,
                                                           httpcore.ConnectTimeout,
                                                           httpcore.PoolTimeout))
                    except ImportError:
                        pass
                if not _is_retryable:
                    try:
                        import openai
                        if isinstance(_exc, openai.APIError):
                            _status = getattr(_exc, "status_code", None)
                            _is_retryable = _status is None or _status >= 500
                    except ImportError:
                        pass
                if not _is_retryable and isinstance(_exc, RuntimeError):
                    _is_retryable = (
                        "response.incomplete" in str(_exc)
                        or "API response truncated" in str(_exc)
                    )

                if _is_retryable and _error_retries < _MAX_ERROR_RETRIES:
                    _error_retries += 1
                    _wait = _RETRY_DELAY * _error_retries
                    _reason = type(_exc).__name__
                    _details = _format_retry_exception(_exc)

                    # If the failure is a max_output_tokens truncation, downgrade
                    # reasoning effort one rung before retrying.  Record the event.
                    _is_truncation = isinstance(_exc, RuntimeError) and (
                        "response.incomplete" in str(_exc)
                        or "API response truncated" in str(_exc)
                    )
                    if _is_truncation:
                        _new_effort = patcher._downgrade_reasoning_effort()
                        if _new_effort is not None:
                            import datetime as _dt
                            _downgrade_log = log_dir / "effort_downgrade.json"
                            _history = []
                            if _downgrade_log.exists():
                                try:
                                    _history = json.loads(_downgrade_log.read_text(encoding="utf-8"))
                                except Exception:
                                    _history = []
                            _history.append({
                                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                                "agent": "patcher",
                                "attempt": _error_retries,
                                "trigger": "max_output_tokens truncation",
                                "new_effort": _new_effort,
                            })
                            _downgrade_log.write_text(
                                json.dumps(_history, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            print(f"{YELLOW}[!] Reasoning effort downgraded → {_new_effort!r} "
                                  f"(logged to {_downgrade_log.name}){RESET}")
                        else:
                            print(f"{YELLOW}[!] Cannot downgrade reasoning effort further "
                                  f"(already at lowest); retrying with current settings.{RESET}")

                    print(
                        f"\n{YELLOW}[!] {_reason} — error retry {_error_retries}/{_MAX_ERROR_RETRIES}"
                        f" — resuming in {_wait}s (session preserved)…{RESET}"
                    )
                    traj_f.write(
                        f"\n\n---\n**[ERROR RETRY {_error_retries}]** {_reason} — resuming after {_wait}s "
                        f"(SQLiteSession carries prior context; no tool re-calls).\n\n"
                        f"```text\n{_details}\n```\n\n"
                    )
                    traj_f.flush()
                    if raw_log_f:
                        raw_log_f.write(
                            f"\n[retry_exception] error_attempt={_error_retries} "
                            f"max={_MAX_ERROR_RETRIES} wait_seconds={_wait} "
                            f"session_resume=True\n"
                            f"{_details}\n"
                        )
                        raw_log_f.flush()
                    printer.flush_thought(agent_instance=patcher)
                    printer.flush_tool()
                    printer.finalize_json()
                    if raw_log_f:
                        raw_log_f.close()
                    traj_f.close()
                    _resume_input_override = None
                    raw_log_f, traj_f = _open_logs(resume=True)
                    printer = StreamPrinter(
                        traj_file=traj_f, raw_log_file=raw_log_f,
                        traj_json_path=log_dir / "skill_patcher_trajectory.json",
                        preload_steps=True,
                    )
                    await asyncio.sleep(_wait)
                else:
                    raise

        printer.flush_thought(agent_instance=patcher)
        printer.flush_tool()
        printer.finalize_json()

        # ── narrative extraction (always, for next iter's lite mode) ─────────
        _traj_json = log_dir / "skill_patcher_trajectory.json"
        _narrative_path = extract_patcher_narrative(_traj_json)
        if _narrative_path:
            print(f"  {GREEN}✓{RESET} patcher narrative: {_narrative_path}")

        # ── cost reporting ────────────────────────────────────────────────────
        from cost_tracker import CostTracker, CostSummary
        summary = CostSummary()
        for s in _streams:
            tracker = CostTracker(model=args.model, label="skill_patcher")
            tracker.observe(s)
            if tracker.run_cost:
                summary.add(tracker.run_cost)
        summary.print_summary(title="Skill Patcher Cost")
        if args.cost_file:
            _cost_resume = _cross_process_resume or _attempt > 0
            summary.save(args.cost_file, resume=_cost_resume)

        _run_ok = patches_complete_path.exists()

    except Exception as exc:
        print(f"\n{RED}[!] Error: {exc}{RESET}")
        import traceback
        traceback.print_exc()
        # Terminal failure path.  Three cases:
        #   1. Azure "Item with id 'rs_...' not found" — stale reasoning-item
        #      reference; preserving session would loop forever.  Delete +
        #      restore skills.
        #   2. Session has accumulated content (>512 B) — preserve for outer
        #      pipeline retry; do NOT restore skills (they reflect committed
        #      progress that the resumed session should keep).
        #   3. Empty session — restore skills + delete (nothing to resume).
        if _is_stale_session_error(exc):
            _restore_file_snapshots(_skill_snapshots)
            try:
                _session_db_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(
                f"{YELLOW}[!] Stale Azure reasoning-item id in session — "
                f"restored skill files and deleted session DB so outer "
                f"retry can start fresh.{RESET}"
            )
        else:
            _session_has_content = (
                _session_db_path.exists() and _session_db_path.stat().st_size > 512
            )
            if _session_has_content:
                print(
                    f"{YELLOW}[!] Session DB intact "
                    f"({_session_db_path.stat().st_size // 1024} KB) — "
                    f"preserving for outer retry checkpoint resume "
                    f"(skill files unchanged).{RESET}"
                )
            else:
                _restore_file_snapshots(_skill_snapshots)
                print(
                    f"{YELLOW}[!] No usable session — restored skill files; "
                    f"outer retry will start fresh.{RESET}"
                )
                try:
                    _session_db_path.unlink(missing_ok=True)
                except Exception:
                    pass
    finally:
        if raw_log_f:
            raw_log_f.close()
        traj_f.close()

    if _run_ok:
        (log_dir / "patcher_complete").touch()

    print(f"\n{BOLD}Artifacts:{RESET}")
    for p in [log_dir / "skill_patcher_trajectory.md",
              log_dir / "skill_patcher_trajectory.json",
              log_dir / "gap_patch_narrative.json",
              log_dir / "gap_patches_complete",
              log_dir / "patcher_complete"]:
        status = GREEN + "✓" + RESET if p.exists() else RED + "✗" + RESET
        print(f"  {status} {p}")

    sys.exit(0 if _run_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
