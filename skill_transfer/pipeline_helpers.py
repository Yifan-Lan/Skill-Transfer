#!/usr/bin/env python3
"""Python worker invoked by run_skill_transfer.sh and run_eval.sh.

Subcommands:
  prepare        Create per-case working dirs from a dataset split.
  run-agents     Run the skill agent over many cases in parallel.
  eval           Score produced outputs against the dataset's ground truth.
  filter         Restrict the batch to the tasks the strong agent solved.
  detect-skills  Aggregate the skill names the agents activated.
  abstract       Turn raw trajectories into typed execution structures.
  validate       Compare weak against strong outputs, per case.

Each subcommand prints its result to stdout (one item per line, or JSON).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
# Benchmark repos live outside this one; override with SPREADSHEETBENCH_DIR / OFFICEQA_DIR.
PYTHON     = sys.executable



# ──────────────────────────────────────────────────────────────────────────────
# prepare
# ──────────────────────────────────────────────────────────────────────────────

def cmd_prepare(args: argparse.Namespace) -> None:
    """Create INSTRUCTION.md + copy init.xlsx for each task."""
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir     = Path(args.out_dir).resolve()

    dataset_file = dataset_dir / "dataset.json"
    if not dataset_file.exists():
        sys.exit(f"ERROR: dataset.json not found in {dataset_dir}")

    tasks: list[dict] = json.loads(dataset_file.read_text(encoding="utf-8"))

    # Filter by split if --split is given.
    # --split          : split name (e.g. "train_20")
    # --split-manifest : explicit path to manifest JSON (default: dataset_dir/split_manifest.json)
    if args.split:
        split_name = args.split
        if getattr(args, "split_manifest", None):
            manifest_file = Path(args.split_manifest)
        else:
            manifest_file = dataset_dir / "split_manifest.json"

        if not manifest_file.exists():
            sys.exit(f"ERROR: split manifest not found: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        split_ids = set(str(i) for i in manifest.get("ids", {}).get(split_name, []))
        if not split_ids:
            sys.exit(f"ERROR: split '{split_name}' not found in {manifest_file} (available: {list(manifest.get('ids', {}).keys())})")
        tasks = [t for t in tasks if str(t.get("id", "")) in split_ids]
        if not tasks:
            sys.exit(f"ERROR: no tasks matched split '{split_name}' IDs in dataset.json")

    if getattr(args, "seed", None) is not None:
        import random
        random.seed(args.seed)
        random.shuffle(tasks)

    if args.simple_only:
        tasks = [t for t in tasks if t.get("instruction_type") == "Cell-Level Manipulation"]
    if args.batch_size:
        tasks = tasks[: args.batch_size]

    prepared: list[str] = []
    for task in tasks:
        task_id = str(task["id"])
        task_type = task.get("type", "spreadsheetbench")
        # WikiTableQuestions: a third dataset family. The instruction lives in
        # the dataset.json `instruction` field (no prompt.txt file), the input
        # workbook is `1_<id>_input.xlsx`, and the answer is a TEXT string in
        # `golden_answer` (no golden xlsx). The agent writes its answer to the
        # cell named by `answer_position` (default Answer!A1) inside output.xlsx.
        is_wtq = task.get("instruction_type") == "WikiTableQuestion"
        src_dir   = dataset_dir / task.get("task_path", task.get("spreadsheet_path", ""))
        case_root = out_dir / "cases" / task_id
        case_dir  = case_root / "case_input"
        case_dir.mkdir(parents=True, exist_ok=True)

        if is_wtq:
            question_text = str(task.get("instruction", "")).strip()
            if not question_text:
                print(f"WARN: empty instruction for WTQ {task_id}", file=sys.stderr)
                continue
        elif task_type in ("dabench", "bird", "searchqa"):
            # DABench/BIRD/SearchQA question lives in `question` (no prompt.txt).
            question_text = str(task.get("question", "")).strip()
            if not question_text:
                print(f"WARN: empty question for {task_type} {task_id}", file=sys.stderr)
                continue
        else:
            # INSTRUCTION.md — accept prompt.txt (SpreadsheetBench/officeqa root tasks/)
            # or INSTRUCTION.md (officeqa split dirs written by prepare_officeqa_dataset.py)
            prompt_path      = src_dir / "prompt.txt"
            instruction_path = src_dir / "INSTRUCTION.md"
            if prompt_path.exists():
                question_text = prompt_path.read_text(encoding="utf-8").strip()
            elif instruction_path.exists():
                question_text = instruction_path.read_text(encoding="utf-8").strip()
            else:
                print(f"WARN: prompt.txt not found for {task_id}", file=sys.stderr)
                continue

        if is_wtq:
            answer_cell = task.get("answer_position", "Answer!A1")
            instruction = (
                question_text
                + f"\n\nThe table is in `Sheet1` of the input workbook. Compute the answer "
                + f"and write ONLY the final answer value into cell `{answer_cell}` "
                + "(create the sheet if needed). Save the workbook as `output.xlsx` in the "
                + "output directory. Write only the answer value — no label, no explanation."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")

            # Copy the input workbook (1_<id>_input.xlsx).
            for cand in (src_dir / f"1_{task_id}_input.xlsx", src_dir / f"{task_id}_input.xlsx"):
                if cand.exists():
                    dst = case_dir / cand.name
                    if not dst.exists():
                        shutil.copy2(cand, dst)
                    break
            else:
                print(f"WARN: input xlsx not found for WTQ {task_id} in {src_dir}", file=sys.stderr)
            # golden_answer stays only in the source dataset.json; it is stripped
            # from the case task_meta below so the agent cannot read the truth.

        elif task_type == "dabench":
            # InfiAgent-DABench: a CSV data-analysis question with a closed-form
            # answer format (@name[value]). The instruction lives in the
            # dataset.json `question` field; the shared CSV is `file_name` under
            # task_path (a shared `tables/` dir). The agent computes the answer
            # and writes the closed-form string to output.txt.
            constraints = str(task.get("constraints", "")).strip()
            fmt         = str(task.get("format", "")).strip()
            fname       = task.get("file_name", "")
            instruction = (
                question_text
                + (f"\n\n## Constraints\n\n{constraints}" if constraints else "")
                + f"\n\n## Data File\n\nThe data is in `{fname}` (a CSV file in the working directory)."
                + f"\n\n## Answer Format\n\nAfter computing the answer, write ONLY the final answer to "
                + f"`output.txt` in the output directory, using EXACTLY this format:\n\n{fmt}\n\n"
                + "Write only the formatted `@name[value]` answer line(s) — no code, no explanation."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")

            src_csv = src_dir / fname
            if src_csv.exists():
                dst = case_dir / fname
                if not dst.exists():
                    shutil.copy2(src_csv, dst)
            else:
                print(f"WARN: CSV not found for dabench {task_id}: {src_csv}", file=sys.stderr)
            # golden_answers stays only in source dataset.json; stripped below.

        elif task_type == "bird":
            # BIRD text-to-SQL: a question over a SQLite DB. The DB is large and
            # shared, so it is NOT copied — the instruction references it by
            # absolute path (read-only exploration) and embeds the schema.
            evidence = str(task.get("evidence", "")).strip()
            schema   = str(task.get("schema", "")).strip()
            db_abs   = (dataset_dir / task.get("task_path", "") / task.get("db_file", "")).resolve()
            no_schema = getattr(args, "no_bird_schema", False)
            if no_schema or not schema:
                # Agentic setup: the agent must discover the schema from the DB itself.
                schema_block = (
                    "\n\n## Database Schema\n\nThe schema is NOT provided. **Inspect the database "
                    "first** to discover its tables, columns, types, and sample values (e.g. query "
                    "`sqlite_master`, or read table info) before writing your query."
                )
            else:
                schema_block = f"\n\n## Database Schema\n\n```sql\n{schema}\n```"
            instruction = (
                question_text
                + (f"\n\n## External Knowledge\n\n{evidence}" if evidence else "")
                + schema_block
                + f"\n\n## Database File\n\nThe SQLite database is at:\n`{db_abs}`\n\n"
                + "You may explore it with **read-only** `SELECT` queries (e.g. via Python's "
                + "`sqlite3`) but must NOT modify it."
                + "\n\n## Output\n\nWrite a SINGLE SQLite `SELECT` query that answers the question "
                + "to `output.sql` in the output directory. Write only the SQL — no markdown fences, "
                + "no explanation."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")
            # gold_sql stays only in source dataset.json; stripped below. DB not copied.

        elif task_type == "searchqa":
            # SearchQA supplies the retrieved web snippets as one context string.
            # Keep the question and context in ordinary case files so the run uses
            # the same file-oriented agent interface as the other benchmarks.
            context = str(task.get("context", "")).strip()
            if not context:
                print(f"WARN: empty context for searchqa {task_id}", file=sys.stderr)
                continue
            (case_dir / "context.txt").write_text(context + "\n", encoding="utf-8")
            instruction = (
                question_text
                + "\n\n## Retrieved Search Context\n\n"
                + "Read `context.txt` in the working directory. Answer the question using only "
                + "the retrieved context; do not use outside knowledge or web access."
                + "\n\n## Output\n\nWrite only the shortest supported final answer to `output.txt` "
                + "in the output directory. Do not include a label, explanation, or citation."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")
            # answers stays only in source dataset.json; stripped below.

        elif task_type == "officeqa":
            source_files = task.get("source_files", [])
            file_list = "\n".join(f"- {f}" for f in source_files)
            instruction = (
                question_text
                + f"\n\n## Source Documents\n\n{file_list}"
                + "\n\nSave your final answer as plain text in `output.txt` in the output directory."
                + "\nWrite only the answer value in `output.txt` — no preamble, no explanation."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")

            # Copy source txt files
            for fname in source_files:
                src_txt = src_dir / fname
                if src_txt.exists():
                    dst = case_dir / fname
                    if not dst.exists():
                        shutil.copy2(src_txt, dst)
                else:
                    print(f"WARN: source txt not found: {src_txt}", file=sys.stderr)

            # golden answer — keep in source dataset only; never copy into case_root
            # so the skill agent (project_root=case_root) cannot read it

        else:
            instruction = (
                question_text
                + "\n\nSave the final modified workbook as `output.xlsx` in the output directory."
            )
            (case_dir / "INSTRUCTION.md").write_text(instruction, encoding="utf-8")

            # Copy init xlsx (handles various SpreadsheetBench formats)
            candidates = [
                src_dir.glob(f"1_{task_id}_init.xlsx"),
                src_dir.glob("initial.xlsx"),
                src_dir.glob(f"1_{task_id}_initial.xlsx"),
                src_dir.glob(f"{task_id}_init.xlsx")
            ]
            for cand_iter in candidates:
                found = list(cand_iter)
                if found:
                    dst = case_dir / found[0].name
                    if not dst.exists():
                        shutil.copy2(found[0], dst)
                    break

        # Task metadata — strip truth fields ("answer" for SSBench, "golden_answer"
        # for WTQ, "golden_answers" for DABench, "answers" for SearchQA) so the skill agent
        # (project_root=case_root) cannot read it.
        task_meta_safe = {k: v for k, v in task.items()
                          if k not in (
                              "answer", "golden_answer", "golden_answers", "gold_sql",
                              "answers", "context",
                          )}
        (case_root / "task_meta.json").write_text(
            json.dumps(task_meta_safe, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (case_root / "strong_outputs").mkdir(exist_ok=True)
        prepared.append(task_id)

    # Output one task_id per line so bash can capture with mapfile
    for tid in prepared:
        print(tid)


# ──────────────────────────────────────────────────────────────────────────────
# run-agents
# ──────────────────────────────────────────────────────────────────────────────

def _run_one(
    case_id: str,
    role: str,
    iter_n: int,
    out_dir: Path,
    model: str,
    skills_dir: Path,
    max_turns: int,
    max_input_tokens_per_turn: int = 0,
    require_skill: bool = False,
    multi_skill: bool = False,
    reasoning_effort: str = "",
    batch_dir: Path | None = None,
    ref_traj_dir: Path | None = None,
    ref_traj_passing_ids: frozenset[str] = frozenset(),
    no_skill: bool = False,
) -> tuple[str, bool]:
    case_root  = out_dir / "cases" / case_id  # global: inputs + strong data
    case_dir   = case_root / "case_input"

    if role == "strong":
        agent_out    = case_root / "strong_outputs"
        traj_path    = case_root / "strong_trajectory.md"
        cost_path    = case_root / "cost_strong.json"
        skills_used  = case_root / "strong_skills_used.json"
    else:
        batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
        iter_dir     = batch_case_root / f"iter_{iter_n:02d}"
        agent_out    = iter_dir / "weak_outputs"
        traj_path    = iter_dir / "weak_trajectory.md"
        cost_path    = iter_dir / "cost_weak.json"
        skills_used  = iter_dir / "weak_skills_used.json"

    agent_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(SCRIPT_DIR / "run_skill_agent.py"),
        "--case-dir",         str(case_dir),
        "--project-root",     str(case_root),
        "--shell-cwd",        str(agent_out),
        "--skills-dir",       str(skills_dir),
        "--model",            model,
        "--out-dir",          str(agent_out),
        "--trajectory-file",  str(traj_path),
        "--cost-file",        str(cost_path),
        "--cost-label",       "strong_skill_agent" if role == "strong" else "weak_skill_agent",
        "--skills-used-file", str(skills_used),
        "--max-turns",        str(max_turns),
        "--mode",             "messages",
    ]
    if max_input_tokens_per_turn > 0:
        cmd += ["--max-input-tokens-per-turn", str(max_input_tokens_per_turn)]
    if require_skill:
        cmd += ["--require-skill"]
    if multi_skill:
        cmd += ["--multi-skill"]
    if no_skill:
        cmd += ["--no-skill"]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if ref_traj_dir is not None and case_id in ref_traj_passing_ids:
        ref_traj_path = ref_traj_dir / "cases" / case_id / "iter_00" / "weak_trajectory.json"
        if ref_traj_path.exists():
            cmd += ["--ref-trajectory", str(ref_traj_path)]
    _MAX_RETRIES = 2
    _TIMEOUT     = 200  # seconds per attempt
    for _attempt in range(_MAX_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()[-300:]
                if err:
                    print(f"  [run-agents] {case_id} exit {result.returncode}: {err}", file=sys.stderr)
                if _attempt < _MAX_RETRIES:
                    print(f"  [run-agents] {case_id} error on attempt {_attempt + 1}/{_MAX_RETRIES + 1} — retrying…", file=sys.stderr)
                    continue
                return case_id, False
            # Sanity check: subprocess returned 0 but the agent may have silently
            # exhausted SDK-internal retries (e.g. Azure 429) without firing any
            # LLM request — produces an empty trajectory and zero-token cost.
            requests_made = -1
            try:
                _cost_data = json.loads(cost_path.read_text(encoding="utf-8"))
                requests_made = int(_cost_data.get("requests", 0))
            except Exception:
                pass
            if requests_made == 0:
                # The agent exits 0 even when the very first LLM call fails, so the
                # real reason is only in its captured output.  Surface it — otherwise
                # a bad API key looks like "no training cases found".
                _out = ((result.stdout or "") + (result.stderr or "")).strip()
                _why = ""
                for _line in _out.splitlines():
                    if "Error" in _line or "error" in _line:
                        _why = _line.strip()[:400]
                        break
                print(
                    f"  [run-agents] {case_id} made 0 LLM requests — the run did nothing.",
                    file=sys.stderr,
                )
                if _why:
                    print(f"  [run-agents] {case_id} reason: {_why}", file=sys.stderr)
                else:
                    print(
                        f"  [run-agents] {case_id} no error text captured "
                        f"— possible silent retry exhaustion in the Agents SDK (rate limit?)",
                        file=sys.stderr,
                    )
                if _attempt < _MAX_RETRIES:
                    print(
                        f"  [run-agents] {case_id} empty-run on attempt {_attempt + 1}/{_MAX_RETRIES + 1} — retrying after 15s…",
                        file=sys.stderr,
                    )
                    time.sleep(15)
                    continue
                print(
                    f"  [run-agents] {case_id} empty-run after {_MAX_RETRIES + 1} attempts — giving up",
                    file=sys.stderr,
                )
                return case_id, False
            return case_id, True
        except subprocess.TimeoutExpired:
            if _attempt < _MAX_RETRIES:
                print(f"  [run-agents] {case_id} timeout on attempt {_attempt + 1}/{_MAX_RETRIES + 1} — retrying…", file=sys.stderr)
            else:
                print(f"  [run-agents] {case_id} timed out after {_MAX_RETRIES + 1} attempts — giving up", file=sys.stderr)
                return case_id, False
        except Exception as exc:
            print(f"  [run-agents] {case_id} failed: {exc}", file=sys.stderr)
            return case_id, False
    return case_id, False


def cmd_run_agents(args: argparse.Namespace) -> None:
    out_dir    = Path(args.out_dir).resolve()
    skills_dir = Path(args.skills_dir).resolve()
    case_ids   = args.case_ids  # list from --case-ids
    batch_dir  = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None

    max_tok          = getattr(args, "max_input_tokens_per_turn", 0)
    require_skill    = getattr(args, "require_skill", False)
    multi_skill      = getattr(args, "multi_skill", False)
    no_skill         = getattr(args, "no_skill", False)
    reasoning_effort = getattr(args, "reasoning_effort", "")
    ref_traj_dir     = Path(args.ref_traj_dir).resolve() if getattr(args, "ref_traj_dir", None) else None
    ref_traj_passing_ids: frozenset[str] = frozenset()
    if ref_traj_dir is not None:
        eval_json = ref_traj_dir / "eval.json"
        if eval_json.exists():
            edata = json.loads(eval_json.read_text())
            ref_traj_passing_ids = frozenset(cid for cid, v in edata.items() if v.get("pass"))
            print(f"  [run-agents] ref-traj: {len(ref_traj_passing_ids)} passing cases will receive reference trajectory", file=sys.stderr)
        else:
            print(f"  [run-agents] ref-traj: eval.json not found in {ref_traj_dir}, injecting for all cases", file=sys.stderr)
            ref_traj_passing_ids = frozenset(cid for cid in case_ids)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_run_one, cid, args.role, args.iter,
                        out_dir, args.model, skills_dir, args.max_turns,
                        max_tok, require_skill, multi_skill, reasoning_effort,
                        batch_dir, ref_traj_dir, ref_traj_passing_ids,
                        no_skill=no_skill): cid
            for cid in case_ids
        }
        for fut in as_completed(futures):
            cid, success = fut.result()
            icon = "✓" if success else "✗"
            # Read per-case cost file
            if args.role == "strong":
                cost_path = out_dir / "cases" / cid / "cost_strong.json"
            else:
                b = batch_dir if batch_dir else out_dir
                cost_path = b / "cases" / cid / f"iter_{args.iter:02d}" / "cost_weak.json"
            cost_str = ""
            try:
                data = json.load(open(cost_path))
                runs = data.get("runs", [data] if "total_cost_usd" in data else [])
                total_in  = sum(r.get("input_tokens",  0) for r in runs)
                total_out = sum(r.get("output_tokens", 0) for r in runs)
                total_cost = sum(r.get("total_cost_usd", 0.0) for r in runs)
                cost_str = f"  [{total_in:,} in + {total_out:,} out = ${total_cost:.4f}]"
            except Exception:
                pass
            print(f"  {icon} {cid} ({args.role} iter={args.iter}){cost_str}", flush=True)


def _load_compare():
    eval_dir = SCRIPT_DIR / "SpreadsheetBench" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from evaluation import compare_workbooks  # type: ignore[import]
    return compare_workbooks


def _load_officeqa_reward():
    reward_dir = Path(os.environ.get("OFFICEQA_DIR") or (REPO_ROOT.parent / "officeqa"))
    if str(reward_dir) not in sys.path:
        sys.path.insert(0, str(reward_dir))
    from reward import fuzzy_match_answer, extract_final_answer  # type: ignore[import]
    return fuzzy_match_answer, extract_final_answer


def _eval_officeqa(
    case_id: str,
    out_dir: Path,
    iter_n: int,  # -1 = strong
    dataset_dir: Path,
    task_path: str,
    batch_dir: Path | None = None,
) -> dict:
    """Evaluate an OfficeQA case using reward.py fuzzy matching at multiple tolerances."""
    zero_scores = {"exact": 0.0, "0.1pct": 0.0, "1pct": 0.0, "5pct": 0.0}

    def fail(message: str) -> dict:
        return {
            "pass": False,
            "score": 0.0,
            "scores_by_tolerance": dict(zero_scores),
            "rationale": message,
            "message": message,
        }

    case_root = out_dir / "cases" / case_id  # global: strong outputs
    golden_file = dataset_dir / task_path / "golden_answer.txt"
    if not golden_file.exists():
        return fail(f"golden_answer.txt not found at {golden_file}")

    if iter_n < 0:
        output_file = case_root / "strong_outputs" / "output.txt"
    else:
        batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
        output_file = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs" / "output.txt"

    if not output_file.exists():
        return fail(f"output.txt not found at {output_file}")

    ground_truth = golden_file.read_text(encoding="utf-8").strip()
    raw_output   = output_file.read_text(encoding="utf-8").strip()

    try:
        fuzzy_match_answer, extract_final_answer = _load_officeqa_reward()
        predicted = extract_final_answer(raw_output)
    except Exception as exc:
        return fail(f"reward.py load error: {exc}")

    # Evaluate at multiple tolerance levels
    tolerances = {"exact": 0.0, "0.1pct": 0.001, "1pct": 0.01, "5pct": 0.05}
    scores_by_tolerance: dict = {}
    rationales: dict = {}
    for name, tol in tolerances.items():
        try:
            is_correct, rationale = fuzzy_match_answer(ground_truth, predicted, tol)
            scores_by_tolerance[name] = 1.0 if is_correct else 0.0
            rationales[name] = rationale
        except Exception as exc:
            scores_by_tolerance[name] = 0.0
            rationales[name] = f"error: {exc}"

    exact_pass = scores_by_tolerance["exact"] == 1.0
    # Primary score = 5% tolerance (most lenient, used for gap analysis)
    primary_score = scores_by_tolerance["5pct"]

    return {
        "pass":                exact_pass,
        "score":               primary_score,
        "scores_by_tolerance": scores_by_tolerance,
        "rationale":           rationales.get("5pct", ""),
        "golden":              ground_truth,
        "output":              predicted,
        "message":             "",
    }


_WTQ_GOLDEN_CACHE: dict[str, dict[str, str]] = {}


def _wtq_golden_map(dataset_dir: Path) -> dict[str, str]:
    """{id -> golden_answer} from the source dataset.json (cached per dataset dir).

    The golden answer is stripped from the per-case task_meta (so the agent
    can't read it), so eval reads it back from the source dataset here."""
    key = str(dataset_dir.resolve())
    if key not in _WTQ_GOLDEN_CACHE:
        m: dict[str, str] = {}
        dj = dataset_dir / "dataset.json"
        if dj.exists():
            for t in json.loads(dj.read_text(encoding="utf-8")):
                if "golden_answer" in t:
                    m[str(t["id"])] = str(t["golden_answer"])
        _WTQ_GOLDEN_CACHE[key] = m
    return _WTQ_GOLDEN_CACHE[key]


_SEARCHQA_GOLDEN_CACHE: dict[str, dict[str, list[str]]] = {}


def _searchqa_golden_map(dataset_dir: Path) -> dict[str, list[str]]:
    """Return SearchQA answer aliases from source dataset.json."""
    key = str(dataset_dir.resolve())
    if key not in _SEARCHQA_GOLDEN_CACHE:
        mapping: dict[str, list[str]] = {}
        dataset_file = dataset_dir / "dataset.json"
        if dataset_file.exists():
            for task in json.loads(dataset_file.read_text(encoding="utf-8")):
                if task.get("type") == "searchqa" and "answers" in task:
                    mapping[str(task["id"])] = [str(a) for a in task["answers"] if str(a).strip()]
        _SEARCHQA_GOLDEN_CACHE[key] = mapping
    return _SEARCHQA_GOLDEN_CACHE[key]


def _normalize_qa_answer(text: str) -> str:
    """SQuAD/MRQA normalization used for SearchQA exact match and token F1."""
    import re  # noqa: PLC0415
    import string  # noqa: PLC0415

    no_punctuation = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def _searchqa_token_f1(prediction: str, golden: str) -> float:
    from collections import Counter  # noqa: PLC0415

    pred_tokens = _normalize_qa_answer(prediction).split()
    gold_tokens = _normalize_qa_answer(golden).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _eval_searchqa(output_file: Path, answers: list[str]) -> dict:
    """MRQA-style normalized exact match and max token F1 over answer aliases."""
    if not output_file.exists():
        return {
            "pass": False,
            "score": 0.0,
            "f1": 0.0,
            "message": f"output.txt not found at {output_file}",
        }
    prediction = output_file.read_text(encoding="utf-8").strip()
    normalized_prediction = _normalize_qa_answer(prediction)
    exact = max(
        (float(normalized_prediction == _normalize_qa_answer(answer)) for answer in answers),
        default=0.0,
    )
    f1 = max((_searchqa_token_f1(prediction, answer) for answer in answers), default=0.0)
    return {
        "pass": exact == 1.0,
        "score": exact,
        "f1": f1,
        "golden": answers,
        "output": prediction,
        "message": "" if exact == 1.0 else f"EM=0, F1={f1:.3f}",
    }


_DABENCH_GOLDEN_CACHE: dict[str, dict[str, list]] = {}


def _dabench_golden_map(dataset_dir: Path) -> dict[str, list]:
    """{id -> [[name, value], ...]} from source dataset.json (cached).

    golden_answers is stripped from the per-case task_meta, so eval reads it
    back from the source dataset here."""
    key = str(dataset_dir.resolve())
    if key not in _DABENCH_GOLDEN_CACHE:
        m: dict[str, list] = {}
        dj = dataset_dir / "dataset.json"
        if dj.exists():
            for t in json.loads(dj.read_text(encoding="utf-8")):
                if "golden_answers" in t:
                    m[str(t["id"])] = t["golden_answers"]
        _DABENCH_GOLDEN_CACHE[key] = m
    return _DABENCH_GOLDEN_CACHE[key]


def _eval_dabench(output_file: Path, golden_pairs: list) -> dict:
    """Closed-form DABench eval — faithful to InfiAgent's eval_closed_form.py.

    Parse @name[value] pairs from output.txt; a sub-answer is correct if it
    string-matches OR |float(pred)-float(gt)| < 1e-6; the question passes iff
    ALL sub-answers are correct (accuracy-by-question)."""
    import re  # noqa: PLC0415
    if not output_file.exists():
        return {"pass": False, "score": 0.0, "message": f"output.txt not found at {output_file}"}
    text = output_file.read_text(encoding="utf-8")
    pred = dict(re.findall(r"@(\w+)\[(.*?)\]", text))  # last wins, as official dict(zip(...))
    label = {str(name): str(val) for name, val in golden_pairs}

    def is_equal(r, l: str) -> bool:
        if r is None:
            return False
        if str(r) == l:
            return True
        try:
            return abs(float(r) - float(l)) < 1e-6
        except Exception:  # noqa: BLE001
            return False

    correctness = {k: is_equal(pred.get(k), label[k]) for k in label}
    passed = bool(correctness) and all(correctness.values())
    n_ok = sum(1 for v in correctness.values() if v)
    return {
        "pass": bool(passed),
        "score": 1.0 if passed else 0.0,
        "sub_score": (n_ok / len(correctness)) if correctness else 0.0,
        "golden": label,
        "output": pred,
        "correctness": correctness,
        "message": "" if passed else f"{n_ok}/{len(correctness)} sub-answers correct",
    }


_BIRD_GOLDEN_CACHE: dict[str, dict[str, dict]] = {}


def _bird_golden_map(dataset_dir: Path) -> dict[str, dict]:
    """{id -> {gold_sql, db_id}} from source dataset.json (cached).

    gold_sql/db_id are stripped from the per-case task_meta, so eval reads
    them back here."""
    key = str(dataset_dir.resolve())
    if key not in _BIRD_GOLDEN_CACHE:
        m: dict[str, dict] = {}
        dj = dataset_dir / "dataset.json"
        if dj.exists():
            for t in json.loads(dj.read_text(encoding="utf-8")):
                if "gold_sql" in t:
                    m[str(t["id"])] = {"gold_sql": t["gold_sql"], "db_id": t.get("db_id", "")}
        _BIRD_GOLDEN_CACHE[key] = m
    return _BIRD_GOLDEN_CACHE[key]


def _run_sql(db_path: str, sql: str, timeout_s: float = 30.0):
    """Execute sql on a sqlite db in a worker thread with a hard timeout.
    Returns (rows_or_None, error_str). rows is a list of tuples."""
    import sqlite3, threading  # noqa: PLC0415
    result: dict = {"rows": None, "err": None}

    def _work():
        try:
            con = sqlite3.connect(db_path, timeout=timeout_s)
            cur = con.cursor()
            cur.execute(sql)
            result["rows"] = cur.fetchall()
            con.close()
        except Exception as exc:  # noqa: BLE001
            result["err"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=_work, daemon=True)
    th.start(); th.join(timeout_s)
    if th.is_alive():
        return None, f"timeout>{timeout_s}s"
    return result["rows"], result["err"]


def _extract_sql(text: str) -> str:
    """Pull a SQL query from the agent's output.sql/output.txt: strip ```sql
    fences and leading prose; take the last statement if several."""
    import re  # noqa: PLC0415
    t = text.strip()
    # prefer fenced ```sql ... ``` block if present
    m = re.findall(r"```(?:sql)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m[-1].strip()
    # drop a leading "SQL:" label
    t = re.sub(r"^\s*sql\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip().rstrip(";").strip()


def _eval_bird(output_file: Path, gold_sql: str, db_path: Path, timeout_s: float = 30.0) -> dict:
    """BIRD execution accuracy: run predicted + gold SQL on the sqlite db and
    compare result-row SETS (order-independent), faithful to BIRD's
    evaluation.py (`set(pred) == set(gold)`)."""
    if not output_file.exists():
        return {"pass": False, "score": 0.0, "message": f"{output_file.name} not found"}
    if not db_path.exists():
        return {"pass": False, "score": 0.0, "message": f"db not found: {db_path}"}
    pred_sql = _extract_sql(output_file.read_text(encoding="utf-8"))
    if not pred_sql:
        return {"pass": False, "score": 0.0, "message": "empty predicted SQL"}

    gold_rows, gold_err = _run_sql(str(db_path), gold_sql, timeout_s)
    if gold_err is not None:
        return {"pass": False, "score": 0.0, "message": f"GOLD sql error (dataset bug?): {gold_err}"}
    pred_rows, pred_err = _run_sql(str(db_path), pred_sql, timeout_s)
    if pred_err is not None:
        return {"pass": False, "score": 0.0, "message": f"pred sql error: {pred_err}",
                "pred_sql": pred_sql}

    ok = set(gold_rows) == set(pred_rows)
    return {
        "pass": bool(ok),
        "score": 1.0 if ok else 0.0,
        "message": "" if ok else f"result mismatch (pred {len(pred_rows)} rows vs gold {len(gold_rows)} rows)",
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
    }


def _eval_wtq(output_file: Path, golden: str, ans_pos: str) -> dict:
    """Read the answer cell from output.xlsx and denotation-match it (official
    WTQ metric via wtq_eval.answers_match)."""
    if not output_file.exists():
        return {"pass": False, "score": 0.0, "message": f"output.xlsx not found at {output_file}"}
    # answer_position is like "Answer!A1"; default sheet=Answer, cell=A1.
    sheet_name, _, cell = ans_pos.partition("!")
    sheet_name = (sheet_name or "Answer").strip("'") or "Answer"
    cell = (cell or "A1").strip() or "A1"
    try:
        from openpyxl import load_workbook  # noqa: PLC0415
        wb = load_workbook(filename=str(output_file), data_only=True)
        if sheet_name not in wb.sheetnames:
            return {"pass": False, "score": 0.0,
                    "message": f"answer sheet '{sheet_name}' not in output.xlsx"}
        raw = wb[sheet_name][cell].value
        predicted = "" if raw is None else str(raw).strip()
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "score": 0.0, "message": f"xlsx read error: {exc}"}

    try:
        import wtq_eval  # noqa: PLC0415
        ok = wtq_eval.answers_match(golden, predicted)
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "score": 0.0, "message": f"wtq_eval error: {exc}"}
    return {"pass": bool(ok), "score": 1.0 if ok else 0.0,
            "golden": golden, "output": predicted, "message": ""}


def _eval_one(
    case_id: str,
    out_dir: Path,
    dataset_dir: Path,
    iter_n: int,  # -1 = strong
    batch_dir: Path | None = None,
) -> dict:
    case_root  = out_dir / "cases" / case_id  # global: task_meta, golden file lookups
    meta_file  = case_root / "task_meta.json"
    if not meta_file.exists():
        return {"pass": False, "message": "task_meta.json not found"}
    task_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # OfficeQA: text answer comparison
    if task_meta.get("type") == "officeqa":
        return _eval_officeqa(case_id, out_dir, iter_n, dataset_dir, task_meta.get("task_path", ""), batch_dir)

    # SearchQA: normalized exact match and token F1 against all answer aliases.
    if task_meta.get("type") == "searchqa":
        if iter_n < 0:
            output_file = case_root / "strong_outputs" / "output.txt"
        else:
            batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
            output_file = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs" / "output.txt"
        answers = _searchqa_golden_map(dataset_dir).get(str(task_meta["id"]))
        if not answers:
            return {"pass": False, "score": 0.0, "f1": 0.0,
                    "message": f"answers for {task_meta['id']} not in dataset.json"}
        return _eval_searchqa(output_file, answers)

    # DABench: closed-form @name[value] answer in output.txt (like officeqa path).
    if task_meta.get("type") == "dabench":
        if iter_n < 0:
            output_file = case_root / "strong_outputs" / "output.txt"
        else:
            batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
            output_file = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs" / "output.txt"
        golden = _dabench_golden_map(dataset_dir).get(str(task_meta["id"]))
        if golden is None:
            return {"pass": False, "message": f"golden_answers for {task_meta['id']} not in dataset.json"}
        return _eval_dabench(output_file, golden)

    # BIRD text-to-SQL: execute predicted output.sql vs gold on the shared DB.
    if task_meta.get("type") == "bird":
        if iter_n < 0:
            base = case_root / "strong_outputs"
        else:
            batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
            base = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs"
        output_file = base / "output.sql"
        if not output_file.exists():
            output_file = base / "output.txt"  # fallback if agent wrote SQL to output.txt
        g = _bird_golden_map(dataset_dir).get(str(task_meta["id"]))
        if not g:
            return {"pass": False, "message": f"gold_sql for {task_meta['id']} not in dataset.json"}
        db_path = dataset_dir / task_meta.get("task_path", "") / task_meta.get("db_file", "")
        return _eval_bird(output_file, g["gold_sql"], db_path)

    task_id    = str(task_meta["id"])
    ans_pos    = task_meta.get("answer_position", "")

    # WikiTableQuestions: read answer cell from output.xlsx, denotation-match the
    # text golden_answer (looked up from the source dataset.json).
    if task_meta.get("instruction_type") == "WikiTableQuestion":
        if iter_n < 0:
            output_file = case_root / "strong_outputs" / "output.xlsx"
        else:
            batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
            output_file = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs" / "output.xlsx"
        golden = _wtq_golden_map(dataset_dir).get(task_id)
        if golden is None:
            return {"pass": False, "message": f"golden_answer for {task_id} not in dataset.json"}
        return _eval_wtq(output_file, golden, ans_pos or "Answer!A1")

    src_dir    = dataset_dir / task_meta["spreadsheet_path"]

    golden_candidates = (
        list(src_dir.glob(f"*_{task_id}_golden.xlsx")) +
        list(src_dir.glob("golden.xlsx")) +
        list(src_dir.glob(f"*_{task_id}_answer.xlsx"))
    )
    if not golden_candidates:
        return {"pass": False, "message": "golden file not found"}
    golden = golden_candidates[0]

    if iter_n < 0:
        output_file = case_root / "strong_outputs" / "output.xlsx"
    else:
        batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
        output_file = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs" / "output.xlsx"

    if not output_file.exists():
        return {"pass": False, "message": f"output.xlsx not found at {output_file}"}

    try:
        compare_workbooks = _load_compare()
        result = compare_workbooks(golden, output_file, task_meta.get("instruction_type", ""), ans_pos)
        passed, msg = result if isinstance(result, tuple) else (bool(result), "")
        return {"pass": passed, "message": msg}
    except Exception as exc:
        return {"pass": False, "message": str(exc)}


def _eval_from_output_dir(
    case_id: str,
    out_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> dict:
    """Evaluate an explicit output directory against ground truth.

    Unlike _eval_one, this takes the exact output dir path rather than
    reconstructing it from iter_n.
    """
    case_root = out_dir / "cases" / case_id
    meta_file = case_root / "task_meta.json"
    if not meta_file.exists():
        return {"pass": False, "message": "task_meta.json not found"}
    task_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    if task_meta.get("type") == "officeqa":
        task_path    = task_meta.get("task_path", "")
        golden_file  = dataset_dir / task_path / "golden_answer.txt"
        if not golden_file.exists():
            return {"pass": False, "message": f"golden_answer.txt not found at {golden_file}"}
        output_file = output_dir / "output.txt"
        if not output_file.exists():
            return {"pass": False, "message": f"output.txt not found at {output_file}"}
        ground_truth = golden_file.read_text(encoding="utf-8").strip()
        raw_output   = output_file.read_text(encoding="utf-8").strip()
        try:
            fuzzy_match_answer, extract_final_answer = _load_officeqa_reward()
            predicted = extract_final_answer(raw_output)
            is_correct, _ = fuzzy_match_answer(ground_truth, predicted, 0.0)
        except Exception as exc:
            return {"pass": False, "message": f"reward.py error: {exc}"}
        return {"pass": bool(is_correct), "message": ""}

    # SearchQA: normalized exact match and token F1.
    if task_meta.get("type") == "searchqa":
        answers = _searchqa_golden_map(dataset_dir).get(str(task_meta["id"]))
        if not answers:
            return {"pass": False, "score": 0.0, "f1": 0.0,
                    "message": f"answers for {task_meta['id']} not in dataset.json"}
        return _eval_searchqa(output_dir / "output.txt", answers)

    # DABench: closed-form @name[value] answer in output.txt.
    if task_meta.get("type") == "dabench":
        golden = _dabench_golden_map(dataset_dir).get(str(task_meta["id"]))
        if golden is None:
            return {"pass": False, "message": f"golden_answers for {task_meta['id']} not in dataset.json"}
        return _eval_dabench(output_dir / "output.txt", golden)

    # BIRD text-to-SQL
    if task_meta.get("type") == "bird":
        output_file = output_dir / "output.sql"
        if not output_file.exists():
            output_file = output_dir / "output.txt"
        g = _bird_golden_map(dataset_dir).get(str(task_meta["id"]))
        if not g:
            return {"pass": False, "message": f"gold_sql for {task_meta['id']} not in dataset.json"}
        db_path = dataset_dir / task_meta.get("task_path", "") / task_meta.get("db_file", "")
        return _eval_bird(output_file, g["gold_sql"], db_path)

    task_id  = str(task_meta["id"])
    ans_pos  = task_meta.get("answer_position", "")

    # WikiTableQuestions: denotation-match the answer cell against text golden.
    if task_meta.get("instruction_type") == "WikiTableQuestion":
        golden = _wtq_golden_map(dataset_dir).get(task_id)
        if golden is None:
            return {"pass": False, "message": f"golden_answer for {task_id} not in dataset.json"}
        return _eval_wtq(output_dir / "output.xlsx", golden, ans_pos or "Answer!A1")

    src_dir  = dataset_dir / task_meta["spreadsheet_path"]

    golden_candidates = (
        list(src_dir.glob(f"*_{task_id}_golden.xlsx")) +
        list(src_dir.glob("golden.xlsx")) +
        list(src_dir.glob(f"*_{task_id}_answer.xlsx"))
    )
    if not golden_candidates:
        return {"pass": False, "message": "golden file not found"}
    golden = golden_candidates[0]

    output_file = output_dir / "output.xlsx"
    if not output_file.exists():
        return {"pass": False, "message": f"output.xlsx not found at {output_file}"}

    try:
        compare_workbooks = _load_compare()
        result = compare_workbooks(golden, output_file, task_meta.get("instruction_type", ""), ans_pos)
        passed, msg = result if isinstance(result, tuple) else (bool(result), "")
        return {"pass": passed, "message": msg}
    except Exception as exc:
        return {"pass": False, "message": str(exc)}


def cmd_eval(args: argparse.Namespace) -> None:
    out_dir     = Path(args.out_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    eval_file   = Path(args.eval_out)
    batch_dir   = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None

    results: dict[str, dict] = {}
    for cid in args.case_ids:
        results[cid] = _eval_one(cid, out_dir, dataset_dir, args.iter, batch_dir)

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results.values() if r["pass"])
    print(f"Eval iter={args.iter}: {passed}/{len(results)} passed → {eval_file}")

    # OfficeQA: also print fine-grained tolerance summary
    officeqa = {
        cid: r for cid, r in results.items()
        if json.loads((out_dir / "cases" / cid / "task_meta.json").read_text(encoding="utf-8")).get("type") == "officeqa"
    }
    if officeqa:
        tol_keys = ["exact", "0.1pct", "1pct", "5pct"]
        parts = []
        for tol in tol_keys:
            n = sum(1 for r in officeqa.values() if r.get("scores_by_tolerance", {}).get(tol, 0) == 1.0)
            parts.append(f"{tol}:{n}/{len(officeqa)}")
        print(f"  OfficeQA ({len(officeqa)} cases) — " + "  ".join(parts))

    searchqa = {
        cid: r for cid, r in results.items()
        if json.loads((out_dir / "cases" / cid / "task_meta.json").read_text(encoding="utf-8")).get("type") == "searchqa"
    }
    if searchqa:
        em = sum(float(r.get("score", 0.0)) for r in searchqa.values()) / len(searchqa)
        f1 = sum(float(r.get("f1", 0.0)) for r in searchqa.values()) / len(searchqa)
        print(f"  SearchQA ({len(searchqa)} cases) — EM:{em:.3f}  F1:{f1:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# filter
# ──────────────────────────────────────────────────────────────────────────────

def cmd_filter(args: argparse.Namespace) -> None:
    """Print task_ids where the strong agent did NOT pass (one per line).

    These cases are excluded from the remaining pool — they are unsolvable
    regardless of skill quality.  Everything else stays in the pool for all
    subsequent iterations in this batch.

    If --case-ids is given, only those case IDs are considered (batch mode).
    """
    strong_eval: dict = json.loads(Path(args.strong_eval).read_text(encoding="utf-8"))

    candidates = args.case_ids if args.case_ids else list(strong_eval.keys())
    remaining = [
        cid for cid in candidates
        if strong_eval.get(cid, {}).get("pass")
    ]
    for cid in sorted(remaining):
        print(cid)


# ──────────────────────────────────────────────────────────────────────────────
# detect-skills
# ──────────────────────────────────────────────────────────────────────────────

def cmd_detect_skills(args: argparse.Namespace) -> None:
    """Print unique skill names (one per line) across agent runs at given iter.

    --role weak  (default): reads iter_NN/weak_skills_used.json
    --role strong          : reads case_root/strong_skills_used.json (iter ignored)

    skills_used.json may be a list of names (legacy) or a dict {name: reason}.
    """
    out_dir   = Path(args.out_dir).resolve()
    batch_dir = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None
    role = getattr(args, "role", "weak")
    skills: set[str] = set()
    for cid in args.case_ids:
        if role == "strong":
            path = out_dir / "cases" / cid / "strong_skills_used.json"
        else:
            b = batch_dir if batch_dir else out_dir
            path = b / "cases" / cid / f"iter_{args.iter:02d}" / "weak_skills_used.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    skills.update(data.keys())
                elif isinstance(data, list):
                    skills.update(data)
            except Exception:
                pass
    for s in sorted(skills):
        print(s)


# ──────────────────────────────────────────────────────────────────────────────
# abstract
# ──────────────────────────────────────────────────────────────────────────────

def _abstract_one(
    case_id: str,
    iter_n: int,
    out_dir: Path,
    model: str,
    structures_out: Path,
    reasoning_effort: str = "",
    batch_dir: Path | None = None,
) -> tuple[str, bool]:
    case_root   = out_dir / "cases" / case_id  # global: inputs + strong data
    task_file   = case_root / "case_input" / "INSTRUCTION.md"
    # Prefer JSON trajectory (richer structure, smaller context) if available
    _s_json = case_root / "strong_trajectory.json"
    _s_md   = case_root / "strong_trajectory.md"
    strong_traj = _s_json if _s_json.exists() else _s_md
    strong_struct = case_root / "strong_structure.json"

    batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
    _w_json = batch_case_root / f"iter_{iter_n:02d}" / "weak_trajectory.json"
    _w_md   = batch_case_root / f"iter_{iter_n:02d}" / "weak_trajectory.md"
    weak_traj   = _w_json if _w_json.exists() else _w_md

    structures_out.mkdir(parents=True, exist_ok=True)
    iter_dir  = batch_case_root / f"iter_{iter_n:02d}"
    cost_file = iter_dir / "cost_abstract.json"
    raw_log   = iter_dir / "abstract_raw_log.txt"

    if iter_n == 0 or not strong_struct.exists():
        cmd = [
            PYTHON, str(SCRIPT_DIR / "run_abstractor.py"),
            "--strong",    str(strong_traj),
            "--weak",      str(weak_traj),
            "--task",      str(task_file),
            "--out",       str(structures_out),
            "--model",     model,
            "--strong-id", "strong",
            "--weak-id",   f"weak_iter{iter_n}",
            "--cost-file", str(cost_file),
            "--raw-log",   str(raw_log),
            "--joint", "--no-viz",
        ]
    else:
        cmd = [
            PYTHON, str(SCRIPT_DIR / "run_abstractor.py"),
            "--strong-structure", str(strong_struct),
            "--strong",           str(strong_traj),
            "--weak",             str(weak_traj),
            "--task",             str(task_file),
            "--out",              str(structures_out),
            "--model",            model,
            "--weak-id",          f"weak_iter{iter_n}",
            "--cost-file",        str(cost_file),
            "--raw-log",          str(raw_log),
            "--no-viz",
        ]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]

    _MAX_RETRIES = 2
    _TIMEOUT     = 200  # seconds per attempt; retry up to _MAX_RETRIES times on timeout or error
    result = None
    for _attempt in range(_MAX_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
            if result.returncode != 0 and _attempt < _MAX_RETRIES:
                print(
                    f"  [abstract] {case_id} error on attempt {_attempt + 1}/{_MAX_RETRIES + 1}"
                    f" — retrying…",
                    file=sys.stderr,
                )
                continue
            break
        except subprocess.TimeoutExpired:
            if _attempt < _MAX_RETRIES:
                print(
                    f"  [abstract] {case_id} timeout on attempt {_attempt + 1}/{_MAX_RETRIES + 1}"
                    f" — retrying…",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [abstract] {case_id} timed out after {_MAX_RETRIES + 1} attempts — giving up",
                    file=sys.stderr,
                )
                return case_id, False
        except Exception as exc:
            print(f"  [abstract] {case_id} failed: {exc}", file=sys.stderr)
            return case_id, False

    try:
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"  [abstract] {case_id} abstractor exited {result.returncode}: {err}", file=sys.stderr)

        # Cache strong structure on first successful abstraction
        if not strong_struct.exists():
            src = structures_out / "strong_structure.json"
            if src.exists():
                shutil.copy2(src, strong_struct)

        # Copy weak structure to iter dir (batch-local)
        weak_src = structures_out / "weak_structure.json"
        weak_dst = batch_case_root / f"iter_{iter_n:02d}" / "weak_structure.json"
        if weak_src.exists():
            weak_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weak_src, weak_dst)

        # Compute and persist structural distance (GED) for history tracking
        if weak_dst.exists() and strong_struct.exists():
            try:
                from execution_structure import ExecutionStructure as _ES  # type: ignore
                from gap_base import compute_structural_distance as _csd   # type: ignore
                s_es = _ES.model_validate(json.loads(strong_struct.read_text(encoding="utf-8")))
                w_es = _ES.model_validate(json.loads(weak_dst.read_text(encoding="utf-8")))
                sd = _csd(s_es, w_es)
                (weak_dst.parent / "structural_distance.json").write_text(
                    json.dumps(sd.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as exc:
                print(f"  [abstract] {case_id} GED computation failed: {exc}", file=sys.stderr)

        return case_id, weak_dst.exists()
    except Exception as exc:
        print(f"  [abstract] {case_id} post-processing failed: {exc}", file=sys.stderr)
        return case_id, False


def cmd_abstract(args: argparse.Namespace) -> None:
    out_dir          = Path(args.out_dir).resolve()
    iter_dir         = Path(args.iter_dir).resolve()
    reasoning_effort = getattr(args, "reasoning_effort", "")
    batch_dir        = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _abstract_one,
                cid, args.iter, out_dir, args.model,
                iter_dir / "structures" / cid,
                reasoning_effort,
                batch_dir,
            ): cid
            for cid in args.case_ids
        }
        for fut in as_completed(futures):
            cid, success = fut.result()
            icon = "✓" if success else "✗"
            print(f"  {icon} {cid} (abstract iter={args.iter})", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# validate
# ──────────────────────────────────────────────────────────────────────────────

def _validate_one(
    case_id: str,
    iter_n: int,
    out_dir: Path,
    model: str,
    dataset_dir: Path | None = None,
    reasoning_effort: str = "",
    batch_dir: Path | None = None,
) -> tuple[str, bool]:
    case_root     = out_dir / "cases" / case_id  # global: strong data, task metadata
    meta_file     = case_root / "task_meta.json"
    strong_struct = case_root / "strong_structure.json"
    strong_out    = case_root / "strong_outputs"
    task_file     = case_root / "case_input" / "INSTRUCTION.md"

    batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
    weak_struct   = batch_case_root / f"iter_{iter_n:02d}" / "weak_structure.json"
    weak_out      = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs"
    validator_dir = batch_case_root / f"iter_{iter_n:02d}" / "validator"

    report_path = validator_dir / "validation_report.json"
    if report_path.exists():
        return case_id, True  # already validated this iteration

    if not strong_struct.exists() or not weak_struct.exists():
        print(f"  [validate] {case_id}: missing structure files — skipping", file=sys.stderr)
        return case_id, False

    cost_file = batch_case_root / f"iter_{iter_n:02d}" / "cost_validator.json"

    cmd = [
        PYTHON, str(SCRIPT_DIR / "run_validator.py"),
        "--strong-structure", str(strong_struct),
        "--weak-structure",   str(weak_struct),
        "--task",             str(task_file),
        "--out",              str(validator_dir),
        "--model",            model,
        "--cost-file",        str(cost_file),
    ]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]

    # OfficeQA: compare against golden_answer.txt (read from source dataset)
    if meta_file.exists():
        task_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if task_meta.get("type") == "officeqa":
            task_path = task_meta.get("task_path", "")
            golden = (dataset_dir / task_path / "golden_answer.txt"
                      if dataset_dir and task_path else None)
            if golden and golden.exists():
                cmd += ["--golden-answer", str(golden)]
            if weak_out.is_dir():
                cmd += ["--weak-results", str(weak_out)]
            return _run_validator_cmd(cmd, case_id, validator_dir)

        # DABench: compare against golden_answers (multi-value @name[value], from
        # source dataset.json — NOT a single scalar, so this must NOT reuse the
        # officeqa fuzzy-tolerance comparator). Materialize the [[name,value],...]
        # pairs as a small JSON file the validator can read; --golden-kind tells
        # run_validator.py/validator_agent.py to build the per-sub-answer exact/
        # 1e-6 comparator instead of compare_answer_fuzzy.
        if task_meta.get("type") == "dabench":
            tid = str(task_meta.get("id", case_id))
            golden_pairs = _dabench_golden_map(dataset_dir).get(tid) if dataset_dir else None
            if golden_pairs:
                validator_dir.mkdir(parents=True, exist_ok=True)
                golden_tmp = validator_dir / "_dabench_golden.json"
                golden_tmp.write_text(json.dumps(golden_pairs, ensure_ascii=False), encoding="utf-8")
                cmd += ["--golden-answer", str(golden_tmp), "--golden-kind", "dabench"]
            if weak_out.is_dir():
                cmd += ["--weak-results", str(weak_out)]
            return _run_validator_cmd(cmd, case_id, validator_dir)

    if strong_out.is_dir():
        cmd += ["--strong-results", str(strong_out)]
    if weak_out.is_dir():
        cmd += ["--weak-results", str(weak_out)]

    return _run_validator_cmd(cmd, case_id, validator_dir)


def _run_validator_cmd(
    cmd: list[str],
    case_id: str,
    validator_dir: Path,
) -> tuple[str, bool]:
    _MAX_RETRIES = 2
    _TIMEOUT     = 200  # seconds per attempt
    for _attempt in range(_MAX_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
            if result.returncode != 0 and _attempt < _MAX_RETRIES:
                err = (result.stderr or result.stdout or "").strip()[-200:]
                print(
                    f"  [validate] {case_id} error on attempt {_attempt + 1}/{_MAX_RETRIES + 1}"
                    f" — retrying…" + (f": {err}" if err else ""),
                    file=sys.stderr,
                )
                continue
            feedback = validator_dir / "validator_feedback.txt"
            return case_id, feedback.exists()
        except subprocess.TimeoutExpired:
            if _attempt < _MAX_RETRIES:
                print(
                    f"  [validate] {case_id} timeout on attempt {_attempt + 1}/{_MAX_RETRIES + 1}"
                    f" — retrying…",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [validate] {case_id} timed out after {_MAX_RETRIES + 1} attempts — giving up",
                    file=sys.stderr,
                )
                return case_id, False
        except Exception as exc:
            print(f"  [validate] {case_id} failed: {exc}", file=sys.stderr)
            return case_id, False
    return case_id, False


def cmd_validate(args: argparse.Namespace) -> None:
    out_dir          = Path(args.out_dir).resolve()
    dataset_dir      = Path(args.dataset_dir).resolve() if getattr(args, "dataset_dir", None) else None
    reasoning_effort = getattr(args, "reasoning_effort", "")
    batch_dir        = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None
    ok_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_validate_one, cid, args.iter, out_dir, args.model, dataset_dir, reasoning_effort, batch_dir): cid
            for cid in args.case_ids
        }
        for fut in as_completed(futures):
            cid, success = fut.result()
            icon = "✓" if success else "✗"
            print(f"  {icon} {cid} (validate iter={args.iter})", flush=True)
            if success:
                ok_count += 1
    print(f"Validated {ok_count}/{len(args.case_ids)} cases → iter {args.iter}")


# ──────────────────────────────────────────────────────────────────────────────
# update-history
# ──────────────────────────────────────────────────────────────────────────────

def cmd_update_history(args: argparse.Namespace) -> None:
    """Update batch/case_history.json with pass, GED, and cell_match for one iteration."""
    batch_dir    = Path(args.batch_dir).resolve()
    out_dir      = Path(args.out_dir).resolve()
    iter_n       = args.iter
    history_path = batch_dir / "case_history.json"

    history: dict = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() \
        else {"updated_at_iter": -1, "cases": {}}

    eval_path = batch_dir / f"weak_eval_iter{iter_n:02d}.json"
    eval_data: dict = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}

    for case_id in args.case_ids:
        # iter-specific data lives in batch_dir/cases/{id}/iter_NN/
        iter_dir = batch_dir / "cases" / case_id / f"iter_{iter_n:02d}"

        # pass/fail from eval
        pass_val = eval_data.get(case_id, {}).get("pass", None)

        # GED from structural_distance.json (written by _abstract_one)
        ged_val: float | None = None
        sd_path = iter_dir / "structural_distance.json"
        if sd_path.exists():
            try:
                sd      = json.loads(sd_path.read_text(encoding="utf-8"))
                ged_val = round(float(sd["normalised_ged"]), 4)
            except Exception:
                pass

        # cell_match_ratio from validation report (first file with a sheet breakdown)
        cell_match: float | None = None
        report_path = iter_dir / "validator" / "validation_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for f in report.get("result_quality", {}).get("files", {}).get("present", []):
                    ratio = f.get("detail", {}).get("cell_match_ratio")
                    if ratio is not None:
                        cell_match = ratio
                        break
            except Exception:
                pass

        entry: dict = {
            "iter":             iter_n,
            "pass":             pass_val,
            "ged_normalised":   ged_val,
            "cell_match_ratio": cell_match,
        }

        # OfficeQA: propagate fuzzy-match scores into history
        fuzzy_scores = eval_data.get(case_id, {}).get("scores_by_tolerance")
        if fuzzy_scores:
            entry["scores_by_tolerance"] = fuzzy_scores

        if case_id not in history["cases"]:
            history["cases"][case_id] = {"iter_history": []}

        # Idempotent: replace any existing entry for this iter
        history["cases"][case_id]["iter_history"] = [
            e for e in history["cases"][case_id]["iter_history"] if e["iter"] != iter_n
        ]
        history["cases"][case_id]["iter_history"].append(entry)
        history["cases"][case_id]["iter_history"].sort(key=lambda e: e["iter"])

    history["updated_at_iter"] = iter_n
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated case history → {history_path}")


# ──────────────────────────────────────────────────────────────────────────────
# recalculate
# ──────────────────────────────────────────────────────────────────────────────

def _recalc_one(case_id: str, out_dir: Path, iter_n: int, batch_dir: Path | None = None) -> tuple[str, bool]:
    """Run LibreOffice recalculation on a single case's weak_outputs dir.
    Skipped automatically for non-workbook task families."""
    case_root = out_dir / "cases" / case_id  # global: task_meta check
    meta_file = case_root / "task_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("type") in ("officeqa", "dabench", "bird", "searchqa"):
            return case_id, True  # no-op for text/SQL-answer tasks

    sys.path.insert(0, str(Path(os.environ.get("SPREADSHEETBENCH_DIR")
                                or (REPO_ROOT.parent / "SpreadsheetBench")) / "evaluation"))
    from open_spreadsheet import detect_backend, find_libreoffice, open_all_spreadsheet_in_dir  # type: ignore[import]

    batch_case_root = (batch_dir if batch_dir else out_dir) / "cases" / case_id
    output_dir = batch_case_root / f"iter_{iter_n:02d}" / "weak_outputs"
    if not output_dir.is_dir():
        return case_id, False

    backend = detect_backend()
    if backend is None:
        print(f"  [recalculate] No backend found for {case_id}", file=sys.stderr)
        return case_id, False

    soffice = find_libreoffice() if backend == "libreoffice" else None
    try:
        open_all_spreadsheet_in_dir(str(output_dir), backend, soffice)
        return case_id, True
    except Exception as e:
        print(f"  [recalculate] Error for {case_id}: {e}", file=sys.stderr)
        return case_id, False


def cmd_recalculate(args: argparse.Namespace) -> None:
    """Open each case's output xlsx in LibreOffice to cache formula values."""
    out_dir   = Path(args.out_dir).resolve()
    batch_dir = Path(args.batch_dir).resolve() if getattr(args, "batch_dir", None) else None
    ok_count  = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_recalc_one, cid, out_dir, args.iter, batch_dir): cid
            for cid in args.case_ids
        }
        for fut in as_completed(futures):
            cid, success = fut.result()
            icon = "✓" if success else "✗"
            print(f"  {icon} {cid} (recalculate iter={args.iter})", flush=True)
            if success:
                ok_count += 1
    print(f"Recalculated {ok_count}/{len(args.case_ids)} cases → iter {args.iter}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Helper for run_skill_transfer.sh")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    p = sub.add_parser("prepare")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-size",  type=int, default=None)
    p.add_argument("--simple-only", action="store_true")
    p.add_argument("--split",          default=None,
                   help="Split name to filter by (e.g. 'train_20').")
    p.add_argument("--split-manifest", default=None,
                   help="Path to split_manifest.json (default: dataset_dir/split_manifest.json).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for shuffling cases before batching. Omit for original dataset order.")
    p.add_argument("--no-bird-schema", action="store_true",
                   help="BIRD only: do NOT embed the DB schema in INSTRUCTION.md; the agent "
                        "must introspect the database itself (agentic setup where a text-to-SQL "
                        "skill's schema-discovery has real value).")

    # run-agents
    p = sub.add_parser("run-agents")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-dir",   default=None,
                   help="Batch output dir (batch_NN/). Weak-agent outputs go here. Omit for single-batch / strong mode.")
    p.add_argument("--model",       required=True)
    p.add_argument("--role",        required=True, choices=["strong", "weak"])
    p.add_argument("--iter",        type=int, default=0)
    p.add_argument("--skills-dir",  required=True)
    p.add_argument("--max-turns",                type=int, default=20)
    p.add_argument("--max-input-tokens-per-turn", type=int, default=0,
                   help="Abort a run if any single turn's input tokens exceed this (0=disabled)")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--case-ids",    nargs="+", required=True)
    p.add_argument("--require-skill", action="store_true", default=False,
                   help="Force agent to call activate_skill at least once")
    p.add_argument("--multi-skill",   action="store_true", default=False,
                   help="Tell agent it may activate more than one skill")
    p.add_argument("--no-skill",      action="store_true", default=False,
                   help="Clean no-skill control: drop the activate_skill tool + skill "
                        "catalog/mandate, and don't refer read_file to a skill for "
                        "xlsx/pdf. Skills under --skills-dir are ignored.")
    p.add_argument("--reasoning-effort", default="",
                   help="Reasoning effort level: 'low', 'medium', or 'high'. Empty = not set.")
    p.add_argument("--ref-traj-dir", default=None,
                   help="Directory whose cases/<id>/iter_00/weak_trajectory.json are used as "
                        "per-case reference trajectories (knowledge-transfer upper bound experiment).")

    # eval
    p = sub.add_parser("eval")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-dir",   default=None,
                   help="Batch output dir (batch_NN/). Weak outputs are read from here. Omit for strong eval.")
    p.add_argument("--iter",        type=int, default=0,
                   help="Iteration index; use -1 for strong agent outputs.")
    p.add_argument("--eval-out",    required=True, help="Path for eval JSON output.")
    p.add_argument("--case-ids",    nargs="+", required=True)

    # filter
    p = sub.add_parser("filter")
    p.add_argument("--strong-eval", required=True)
    p.add_argument("--weak-eval",   required=True)
    p.add_argument("--case-ids",    nargs="+", default=None,
                   help="Restrict filtering to these case IDs (batch mode).")

    # detect-skills
    p = sub.add_parser("detect-skills")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--batch-dir",  default=None,
                   help="Batch output dir (batch_NN/). Used for weak role skill detection.")
    p.add_argument("--iter",       type=int, default=0)
    p.add_argument("--role",       default="weak", choices=["weak", "strong"])
    p.add_argument("--case-ids",   nargs="+", required=True)

    # abstract
    p = sub.add_parser("abstract")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-dir",   default=None,
                   help="Batch output dir (batch_NN/). Weak trajectories and structures go here.")
    p.add_argument("--iter-dir",    required=True)
    p.add_argument("--iter",        type=int, default=0)
    p.add_argument("--model",       required=True)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--case-ids",    nargs="+", required=True)
    p.add_argument("--reasoning-effort", default="",
                   help="Reasoning effort level: 'low', 'medium', or 'high'. Empty = not set.")

    # validate
    p = sub.add_parser("validate")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-dir",   default=None,
                   help="Batch output dir (batch_NN/). Weak structures and validator reports are here.")
    p.add_argument("--iter",        type=int, default=0)
    p.add_argument("--model",       default="gpt-5.4")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--case-ids",    nargs="+", required=True)
    p.add_argument("--dataset-dir", default=None, help="Source dataset dir (required for OfficeQA golden answer lookup)")
    p.add_argument("--reasoning-effort", default="",
                   help="Reasoning effort level: 'low', 'medium', or 'high'. Empty = not set.")

    # recalculate
    p = sub.add_parser("recalculate")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--batch-dir",   default=None,
                   help="Batch output dir (batch_NN/). Weak outputs are recalculated here.")
    p.add_argument("--iter",        type=int, default=0)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--case-ids",    nargs="+", required=True)

    # update-history
    p = sub.add_parser("update-history")
    p.add_argument("--batch-dir", required=True, help="Batch output directory (contains weak_eval_iterNN.json).")
    p.add_argument("--out-dir",   required=True, help="Pipeline output root (contains cases/<id>/).")
    p.add_argument("--iter",      type=int, required=True)
    p.add_argument("--case-ids",  nargs="+", required=True)

    args = parser.parse_args()
    {
        "prepare":              cmd_prepare,
        "run-agents":           cmd_run_agents,
        "eval":                 cmd_eval,
        "filter":               cmd_filter,
        "detect-skills":        cmd_detect_skills,
        "abstract":             cmd_abstract,
        "validate":             cmd_validate,
        "recalculate":          cmd_recalculate,
        "update-history":       cmd_update_history,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
