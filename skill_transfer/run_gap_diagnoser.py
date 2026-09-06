#!/usr/bin/env python3
"""Command-line runner for GapDiagnoser.

Reads the strong/weak execution structures of a batch, runs the diagnosis, and
writes gap_report.json plus a trajectory of the run.  Retries on API errors and
on text-only early stops, and can resume from its session database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from openai_setup import setup_openai
setup_openai("gap-diagnoser")

# Shared helpers: color constants, retry exception formatter, log-archive
# helper, file snapshot helpers, and StreamPrinter.  Keeping them in one
# place guarantees consistent behavior across the two agents.
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


async def main():
    parser = argparse.ArgumentParser(
        description="Run GapDiagnoser: analyse structural gaps and emit gap_report.json "
                    "for downstream consumption by SkillPatcher.",
    )
    # ── case selection (multi-mode only — Diagnoser is multi-case by design) ──
    parser.add_argument(
        "--cases-dir", required=True,
        help="Directory containing per-case subdirs with weak-specific data "
             "(iter_<N>/weak_structure.json, iter_<N>/weak_trajectory.*).",
    )
    parser.add_argument(
        "--global-cases-dir", default="",
        help="Directory containing global per-case subdirs with strong data "
             "(strong_structure.json, strong_trajectory.*). "
             "Defaults to --cases-dir when not provided.",
    )
    parser.add_argument(
        "--iter", type=int, default=0,
        help="Which iteration's weak structures/trajectories to use (default: 0).",
    )
    # ── skills ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--skill-dir", default=str(PROJECT_ROOT / "skills"),
        help="Root directory of the working skills library.",
    )
    parser.add_argument(
        "--skill-names", nargs="+", required=True,
        help="One or more working skill names (e.g. 'xlsx' or 'pdf xlsx'). "
             "These are the skills SkillPatcher will modify based on gap_report.json.",
    )
    parser.add_argument(
        "--meta-skills-dir", default=str(PROJECT_ROOT / "skills" / "meta_brainstorming"),
        help="Directory of meta-skills (default: brainstorming).  Activated via "
             "the activate_skill tool to provide diagnostic-diversity guidance.",
    )
    # ── previous-iteration evidence ───────────────────────────────────────────
    parser.add_argument(
        "--prev-gap-trajectory-files", nargs="+", default=[],
        help="Paths to all previous iterations' gap diagnoser/agent trajectory files in "
             "chronological order.  Used in STEP 3 SYNTHESIS for cross-iter status.",
    )
    parser.add_argument(
        "--prev-gap-patch-narrative-files", nargs="+", default=[],
        help="Lite alternative to --prev-gap-trajectory-files.  Paths to "
             "gap_patch_narrative.json files (one per previous iteration, chronological). "
             "Each file contains only the patch-phase reasoning and "
             "replace_in_file/apply_patch calls SkillPatcher made.  The corresponding "
             "gap_report.json is resolved automatically from the same directory. "
             "Mutually exclusive with --prev-gap-trajectory-files.",
    )
    # ── model / output ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", default="gpt-5.4",
        help="Model for the Diagnoser agent.",
    )
    parser.add_argument(
        "--project-root", default=str(PROJECT_ROOT),
        help="Project root for file-path sandboxing.",
    )
    parser.add_argument(
        "--log-dir", default="",
        help="Directory to write raw_log.txt, trajectory.md/.json, gap_report.json, "
             "diagnoser_complete sentinel, and the SQLite session DB (default: cases-dir).",
    )
    parser.add_argument(
        "--no-raw-log", action="store_true",
        help="Skip writing the raw event log.",
    )
    parser.add_argument(
        "--cost-file", default="",
        help="If set, write a JSON cost summary for this run to this path.",
    )
    parser.add_argument(
        "--force-read-trajectories", action="store_true", default=False,
        help="Force the diagnoser to read both strong and weak trajectory files "
             "before analysis (default: read only when code_snippets are insufficient).",
    )
    parser.add_argument(
        "--trajectory-format", default="json", choices=["md", "json"],
        help="Format of trajectory files exposed to the diagnoser via read_case_trajectory.",
    )
    parser.add_argument(
        "--instruction-save-path", default="",
        help="If set, save the full prompt sent to the diagnoser to this file.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=50,
        help="Maximum agent turns (default: 50).",
    )
    parser.add_argument(
        "--case-history-file", default="",
        help="Path to batch case_history.json written by pipeline_helpers.  Read as "
             "the mandatory first step in STEP 0.",
    )
    parser.add_argument("--min-gap-count", type=int, default=1,
                        help="Minimum failed+pass case count for a gap to enter "
                             "recommended_patch_order (default: 1 — admit single-case "
                             "gaps; raise to 2 for stricter filtering)")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--max-output-tokens", type=int, default=0)
    args = parser.parse_args()

    # ── mode-independent setup ────────────────────────────────────────────────
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

    # ── build case contexts ───────────────────────────────────────────────────
    cases_root = Path(args.cases_dir).resolve()
    global_cases_root = (
        Path(args.global_cases_dir).resolve()
        if args.global_cases_dir else cases_root
    )
    iter_n = args.iter
    log_dir = Path(args.log_dir).resolve() if args.log_dir else cases_root
    log_dir.mkdir(parents=True, exist_ok=True)

    from gap_diagnoser import GapDiagnoser
    from gap_base import CaseContext

    case_contexts = []
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        global_case_dir = global_cases_root / case_dir.name
        s_struct = global_case_dir / "strong_structure.json"
        w_struct = case_dir / f"iter_{iter_n:02d}" / "weak_structure.json"
        _s_json = global_case_dir / "strong_trajectory.json"
        s_traj  = _s_json if _s_json.exists() else global_case_dir / "strong_trajectory.md"
        _w_json = case_dir / f"iter_{iter_n:02d}" / "weak_trajectory.json"
        w_traj  = _w_json if _w_json.exists() else case_dir / f"iter_{iter_n:02d}" / "weak_trajectory.md"
        if s_struct.exists() and w_struct.exists():
            case_contexts.append(CaseContext(
                case_id=case_dir.name,
                strong_structure_path=str(s_struct),
                weak_structure_path=str(w_struct),
                strong_traj_path=str(s_traj),
                weak_traj_path=str(w_traj),
            ))

    if not case_contexts:
        print(f"{RED}[!] No valid cases found in {cases_root}{RESET}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{BOLD}Gap Diagnoser{RESET}")
    print(f"  model              : {args.model}")
    print(f"  cases              : {len(case_contexts)} from {cases_root}")
    print(f"  iteration          : {iter_n}")
    print(f"  log dir            : {log_dir}")
    print(f"  meta-skills        : "
          f"{args.meta_skills_dir}")
    for sf in skill_files:
        print(f"  working skill      : {sf}")

    # ── SQLiteSession with distinct session_id ────────────────────────────────
    from agents.memory.sqlite_session import SQLiteSession
    _session_db_path = log_dir / "diagnoser_session.sqlite"
    _cross_process_resume = (
        _session_db_path.exists() and _session_db_path.stat().st_size > 512
    )
    if _cross_process_resume:
        print(f"{YELLOW}[session] Existing diagnoser session DB found — will resume prior run "
              f"({_session_db_path.stat().st_size // 1024} KB).{RESET}")
    _session = SQLiteSession(session_id="gap-diagnoser-main",
                             db_path=str(_session_db_path))

    def _open_logs(resume: bool = False):
        import datetime as _dt
        mode = "a" if resume else "w"
        raw = (
            None if args.no_raw_log
            else (log_dir / "gap_diagnoser_raw_log.txt").open(mode, encoding="utf-8")
        )
        traj = (log_dir / "gap_diagnoser_trajectory.md").open(mode, encoding="utf-8")
        if resume:
            traj.write(f"\n\n---\n**[RESUMED]** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            traj.write(
                f"# Gap Diagnoser Trajectory\n\n"
                f"**model**: {args.model}  \n"
                f"**cases**: {len(case_contexts)}  \n"
                f"**meta-skills**: {args.meta_skills_dir}\n\n"
            )
        return raw, traj

    raw_log_f, traj_f = _open_logs(resume=_cross_process_resume)

    # ── build the agent ───────────────────────────────────────────────────────
    diagnoser = GapDiagnoser(
        model=args.model,
        project_root=project_root,
        task_dir=cases_root,
        diff_log_path=None,  # Diagnoser doesn't edit skills
        trajectory_format=args.trajectory_format,
        min_gap_count=args.min_gap_count,
        model_kwargs=_model_kwargs,
        meta_skills_dir=args.meta_skills_dir,
    )

    # Save the assembled system prompt to the log dir for audit / debugging.
    # Overwritten on every run (including resumes) — the prompt is built
    # deterministically from CLI args, so the on-disk copy always matches
    # what the live agent is using.
    try:
        (log_dir / "gap_diagnoser_system_prompt.md").write_text(
            diagnoser.system_prompt_multi, encoding="utf-8"
        )
    except Exception as exc:
        print(f"{YELLOW}[!] Failed to save system prompt: {exc}{RESET}", file=sys.stderr)

    printer = StreamPrinter(
        traj_file=traj_f,
        raw_log_file=raw_log_f,
        traj_json_path=log_dir / "gap_diagnoser_trajectory.json",
        preload_steps=_cross_process_resume,
    )

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: (
        print(f"\n\n{RED}[!] Interrupted.{RESET}"), loop.stop()
    ))

    print(f"\n{BOLD}Starting diagnosis…{RESET}\n{'─' * 60}")

    # ── prev-iteration evidence resolution ────────────────────────────────────
    prev_gap_trajectory_paths = [
        str(Path(p).resolve())
        for p in args.prev_gap_trajectory_files
        if Path(p).exists()
    ]
    prev_gap_patch_narrative_paths = [
        str(Path(p).resolve())
        for p in args.prev_gap_patch_narrative_files
        if Path(p).exists()
    ]
    if prev_gap_patch_narrative_paths:
        prev_gap_trajectory_paths = []  # narrative mode takes precedence

    instruction_save_path: str | None = None
    if args.instruction_save_path:
        p = Path(args.instruction_save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        instruction_save_path = str(p.resolve())

    case_history_path: str | None = None
    if args.case_history_file:
        p = Path(args.case_history_file)
        if p.exists():
            case_history_path = str(p.resolve())

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

    gap_report_path = log_dir / "gap_report.json"

    try:
        for _attempt in range(_MAX_ERROR_RETRIES + _MAX_TEXT_ONLY_RETRIES + 1):
            _resuming = _attempt > 0 or _cross_process_resume
            try:
                stream = diagnoser.run_streamed_diagnoser(
                    task_description=(
                        f"Multi-case gap diagnosis over {len(case_contexts)} "
                        f"training cases.  Submit a gap_report.json whose "
                        f"skill_patch_hint per gap is a 200–400 word blueprint."
                    ),
                    cases=case_contexts,
                    skill_paths=[str(sf) for sf in skill_files],
                    out_dir=str(log_dir),
                    force_read_trajectories=args.force_read_trajectories,
                    prev_gap_trajectory_paths=prev_gap_trajectory_paths or None,
                    prev_gap_patch_narrative_paths=prev_gap_patch_narrative_paths or None,
                    instruction_save_path=instruction_save_path,
                    case_history_path=case_history_path,
                    max_turns=args.max_turns,
                    session=_session,
                    resuming=_resuming,
                    resume_input=_resume_input_override,
                )
                _streams.append(stream)

                _stream_incomplete = False
                async for event in stream.stream_events():
                    printer.handle(event, agent_instance=diagnoser)
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

                # Detect text-only termination: the stream ended cleanly but
                # the diagnosis is incomplete.  The Diagnoser is done once
                # gap_report.json exists, so its absence means the agent
                # stopped somewhere in STEPs 0–5; retry with the resume cue.
                _gap_report_exists = gap_report_path.exists()
                _text_only_stop = (
                    not _gap_report_exists
                    and _text_only_retries < _MAX_TEXT_ONLY_RETRIES
                )
                if _text_only_stop:
                    _text_only_retries += 1
                    _wait = _TEXT_ONLY_DELAY
                    _resume_input_override = diagnoser._TEXT_ONLY_PRE_REPORT_RESUME_INPUT
                    print(
                        f"\n{YELLOW}[!] Diagnoser stopped early (text-only termination, pre-report) "
                        f"— text-only retry {_text_only_retries}/{_MAX_TEXT_ONLY_RETRIES} "
                        f"in {_wait}s (session preserved)…{RESET}"
                    )
                    traj_f.write(
                        f"\n\n---\n**[TEXT-ONLY RETRY {_text_only_retries}]** pre-report termination — "
                        f"resuming after {_wait}s (SQLiteSession carries prior work).\n\n"
                    )
                    traj_f.flush()
                    if raw_log_f:
                        raw_log_f.write(
                            f"\n[retry_text_only] text_only_attempt={_text_only_retries} "
                            f"max={_MAX_TEXT_ONLY_RETRIES} phase=pre-report wait_seconds={_wait} "
                            f"session_resume=True\n"
                        )
                        raw_log_f.flush()
                    printer.flush_thought(agent_instance=diagnoser)
                    printer.flush_tool()
                    printer.finalize_json()
                    if raw_log_f:
                        raw_log_f.close()
                    traj_f.close()
                    raw_log_f, traj_f = _open_logs(resume=True)
                    printer = StreamPrinter(
                        traj_file=traj_f, raw_log_file=raw_log_f,
                        traj_json_path=log_dir / "gap_diagnoser_trajectory.json",
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
                    # reasoning effort one rung before retrying.  Record the event
                    # so the pipeline can summarise at end.
                    _is_truncation = isinstance(_exc, RuntimeError) and (
                        "response.incomplete" in str(_exc)
                        or "API response truncated" in str(_exc)
                    )
                    if _is_truncation:
                        _new_effort = diagnoser._downgrade_reasoning_effort()
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
                                "agent": "diagnoser",
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
                    printer.flush_thought(agent_instance=diagnoser)
                    printer.flush_tool()
                    printer.finalize_json()
                    if raw_log_f:
                        raw_log_f.close()
                    traj_f.close()
                    _resume_input_override = None
                    raw_log_f, traj_f = _open_logs(resume=True)
                    printer = StreamPrinter(
                        traj_file=traj_f, raw_log_file=raw_log_f,
                        traj_json_path=log_dir / "gap_diagnoser_trajectory.json",
                        preload_steps=True,
                    )
                    await asyncio.sleep(_wait)
                else:
                    raise

        printer.flush_thought(agent_instance=diagnoser)
        printer.flush_tool()
        printer.finalize_json()

        # ── cost reporting ────────────────────────────────────────────────────
        from cost_tracker import CostTracker, CostSummary
        summary = CostSummary()
        for s in _streams:
            tracker = CostTracker(model=args.model, label="gap_diagnoser")
            tracker.observe(s)
            if tracker.run_cost:
                summary.add(tracker.run_cost)
        summary.print_summary(title="Gap Diagnoser Cost")
        if args.cost_file:
            _cost_resume = _cross_process_resume or _attempt > 0
            summary.save(args.cost_file, resume=_cost_resume)

        _run_ok = gap_report_path.exists()

    except Exception as exc:
        print(f"\n{RED}[!] Error: {exc}{RESET}")
        import traceback
        traceback.print_exc()
        # Terminal failure path.  Three cases:
        #   1. Azure "Item with id 'rs_...' not found" — session DB
        #      references a reasoning item Azure has GC'd; preserving
        #      it would loop forever.  Delete and start fresh.
        #   2. Session has accumulated content (>512 B) — preserve for
        #      outer pipeline retry to resume from checkpoint.
        #   3. Empty session — delete (nothing to resume from).
        if _is_stale_session_error(exc):
            try:
                _session_db_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(
                f"{YELLOW}[!] Stale Azure reasoning-item id in session — "
                f"deleting session DB so outer retry can start fresh.{RESET}"
            )
        else:
            _session_has_content = (
                _session_db_path.exists() and _session_db_path.stat().st_size > 512
            )
            if _session_has_content:
                print(
                    f"{YELLOW}[!] Session DB intact "
                    f"({_session_db_path.stat().st_size // 1024} KB) — "
                    f"preserving for outer retry checkpoint resume.{RESET}"
                )
            else:
                try:
                    _session_db_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(
                    f"{YELLOW}[!] No usable session — outer retry will start fresh.{RESET}"
                )
    finally:
        if raw_log_f:
            raw_log_f.close()
        traj_f.close()

    if _run_ok:
        (log_dir / "diagnoser_complete").touch()

    print(f"\n{BOLD}Artifacts:{RESET}")
    for p in [gap_report_path,
              log_dir / "gap_diagnoser_trajectory.md",
              log_dir / "gap_diagnoser_trajectory.json",
              log_dir / "diagnoser_complete"]:
        status = GREEN + "✓" + RESET if p.exists() else RED + "✗" + RESET
        print(f"  {status} {p}")

    sys.exit(0 if _run_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
