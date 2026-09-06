"""Validator: per-case comparison of weak output against strong output.

Produces a structured report (file presence, content match, per-column match)
that the Diagnoser reads alongside the execution structures.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agents import Agent, Runner, function_tool

from validator import (
    _match_files_by_content,
    _file_fingerprint,
    compare_file,
)

logger = logging.getLogger(__name__)

# Suffixes that are always intermediate execution artifacts, never final outputs.
# Applied consistently to both strong and weak file listings so scripts/logs
# never pollute the expected-output set or the false-positive count.
_INTERMEDIATE_SUFFIXES = {".py", ".pyc", ".sh", ".bash", ".log", ".tmp"}


def _is_output_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() not in _INTERMEDIATE_SUFFIXES


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

VALIDATOR_AGENT_SYSTEM_PROMPT = """
You are a Validation Agent for a skill-distillation pipeline.

Your job: determine whether the weak agent produced correct output for the task,
by reading and reasoning about the actual file contents.

## Intermediate / execution files — NEVER count as expected outputs

Files with extensions .py, .sh, .bash, .pyc, .log, .tmp are execution artifacts
produced during the agent's run. They are NEVER final outputs the user requested.
Do NOT list them in missing_from_weak, present, or extra_in_weak — ignore them entirely.
When `strong_produced_files` is provided in the prompt, treat that list as the ONLY
expected outputs. Anything not on that list is not an expected deliverable.

## Your role as a judge

Programmatic tools (find_content_matches, compare_files) are HELPERS that save you
reading time — they are NOT ground truth.  Their results can be wrong:

  • Fingerprint matching misses semantically identical files with minor formatting
    differences (e.g. multi-line cells vs single-line, extra whitespace).
  • Row-match stats may report 0% for a file that is actually correct because of
    cell-level whitespace differences.
  • A file reported as "missing" may in fact be present under a different name or
    with garbled extraction — it was attempted, just done poorly.

You MUST read the actual file contents and form your own judgment.
Do not report a file as incorrect just because a programmatic tool says so.
Do not report a file as missing just because no fingerprint match was found.

## Workflow

1. list_files(strong_dir) and list_files(weak_dir) to see what exists
2. find_content_matches(strong_dir, weak_dir) as a first-pass hint
3. For every strong reference file:
   a. If fingerprint-matched to a weak file → call compare_files, then read_file
      on both to verify the comparison result with your own eyes
   b. If NOT fingerprint-matched (reported "missing") → read_file on the strong
      reference, then read_file on each unmatched weak file to check if any is
      actually the same output produced with errors. To qualify as PARTIAL, the
      weak file MUST show clear evidence of being an attempt at the same output:
      overlapping key entities, data fields, or subject matter that correspond to
      the strong reference. If the weak file's content is unrelated — different
      subject, different source, or clearly off-topic — it is NOT a match;
      classify the strong file as MISSING and the weak file as a FALSE POSITIVE.
      Garbled or corrupted structure alone is not sufficient for PARTIAL — there
      must be shared subject matter between the weak file and the strong reference.
4. Classify each strong file as:
   - CORRECT : data fully matches (after accounting for minor formatting)
   - PARTIAL : present and clearly an attempt at the same output — overlapping subject matter, key entities, or data fields match the strong reference — but with extraction/generation errors (corrupted structure, missing data)
   - MISSING : weak agent produced nothing for this expected output at all
5. Classify each unmatched weak file as:
   - FALSE POSITIVE: content clearly unrelated to any expected output
6. Compute your own scores based on your classification:
   - file_presence_ratio = (CORRECT + PARTIAL) / total_expected
   - content_match_ratio = CORRECT / (CORRECT + PARTIAL)  [0.0 if file_presence_ratio = 0; 1.0 if CORRECT > 0 and PARTIAL = 0]
   - overall_quality_score = 0.4 * file_presence_ratio + 0.6 * content_match_ratio
7. Determine verdict:
   - PASS    : all expected files CORRECT, no false positives
   - PARTIAL : some files PARTIAL or one expected file MISSING
   - FAIL    : most expected files MISSING or severely wrong
8. write_file → validation_report.json
9. write_file → validator_feedback.txt

## validation_report.json schema

{
  "strong_agent": "<id>",
  "weak_agent": "<id>",
  "verdict": "PASS|PARTIAL|FAIL",
  "result_quality": {
    "files": {
      "present": [
        {
          "filename": "<strong_filename>",
          "matched_weak_filename": "<weak_filename, same if identical>",
          "content_match": true|false,
          "comparator": "tabular|json|text|bytes",
          "detail": { ... comparator-specific stats from compare_files ... }
        }
      ],
      "missing_from_weak": ["<filename>", ...],
      "extra_in_weak": ["<filename>", ...]
    },
    "scores": {
      "file_presence_ratio": 0.0,
      "content_match_ratio": 0.0,
      "overall_quality_score": 0.0
    }
  },
  "analysis": "<your reading-based interpretation: what is correct, what is wrong, why>",
  "verdict_reasoning": "<how you determined the verdict from your classification>"
}

## validator_feedback.txt format

Always include the first three lines. Only include the remaining lines when applicable
(omit lines that do not apply — do NOT write them with empty values):

Verdict      : PASS|PARTIAL|FAIL
File presence: <ratio>
Content match: <ratio>
Missing output files (weak did not produce): ['<file>', ...]
  Reference content of missing files (what weak should have produced):
    <filename>: <brief content preview — first few lines or key fields>
Partially correct files (present but content errors): ['<file>', ...]
  Details of errors:
    <filename>: <description of what is wrong vs reference>
  Fine-grained metrics (include when compare_files returns tabular/xlsx stats):
    <filename>: cell_match=X.XX  row_match=X.XX  per_column={col: X.XX, ...}
Extra output files (weak over-produced, likely false positives): ['<file>', ...]
  Content of extra files (evidence for why they should be excluded):
    <filename>: <brief content preview — first few lines or key fields>
Analysis: <root cause and what skill steps need to change so the weak agent does better>
"""


# ──────────────────────────────────────────────────────────────────────────────
# ValidatorAgent
# ──────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """Hybrid programmatic + LLM validator agent."""

    def __init__(self, model: str = "gpt-5.4", model_kwargs: dict | None = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self._stream_results: list = []

    def validate(
        self,
        task: str,
        strong_dir: Path,
        weak_dir: Path,
        strong_agent_id: str,
        weak_agent_id: str,
        strong_produced_files: list[str],
        out_dir: Path,
        *,
        weak_produced_files: list[str] | None = None,
        golden_answer_file: Path | None = None,
        golden_answer_kind: str = "officeqa",
        report_filename: str = "validation_report.json",
        feedback_filename: str = "validator_feedback.txt",
        on_event=None,
    ) -> dict:
        """
        Run the validator agent and return the parsed validation_report dict.

        Writes <out_dir>/<report_filename> and <out_dir>/<feedback_filename>.
        Pass on_event=callable to receive streaming events (e.g. StreamPrinter.handle).
        weak_produced_files: if provided, only these files from weak_dir are considered
        (intermediate artifacts outside this list are ignored).
        golden_answer_kind: "officeqa" (golden_answer_file is a single scalar text answer,
        matched via fuzzy tolerance) or "dabench" (golden_answer_file is a JSON
        [[name,value],...] list, matched per-sub-answer via exact-string-or-1e-6, ALL must
        match). Ignored when golden_answer_file is None.
        """
        return asyncio.run(self._validate_async(
            task=task,
            strong_dir=strong_dir,
            weak_dir=weak_dir,
            strong_agent_id=strong_agent_id,
            weak_agent_id=weak_agent_id,
            strong_produced_files=strong_produced_files,
            weak_produced_files=weak_produced_files,
            golden_answer_file=golden_answer_file,
            golden_answer_kind=golden_answer_kind,
            out_dir=out_dir,
            report_filename=report_filename,
            feedback_filename=feedback_filename,
            on_event=on_event,
        ))

    async def _validate_async(
        self,
        task: str,
        strong_dir: Path,
        weak_dir: Path,
        strong_agent_id: str,
        weak_agent_id: str,
        strong_produced_files: list[str],
        out_dir: Path,
        report_filename: str,
        feedback_filename: str,
        weak_produced_files: list[str] | None = None,
        golden_answer_file: Path | None = None,
        golden_answer_kind: str = "officeqa",
        on_event=None,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path   = out_dir / report_filename
        feedback_path = out_dir / feedback_filename

        # Strip execution artifacts from expected file lists (basenames only)
        def _strip_intermediates(files: list[str]) -> list[str]:
            return [
                Path(f).name for f in files
                if Path(f).suffix.lower() not in _INTERMEDIATE_SUFFIXES
            ]

        strong_produced_files = _strip_intermediates(strong_produced_files)
        if weak_produced_files is not None:
            weak_produced_files = _strip_intermediates(weak_produced_files)

        tools = self._build_tools(strong_dir, weak_dir, out_dir, strong_produced_files,
                                   weak_produced_files, golden_answer_file, golden_answer_kind)
        agent = Agent(
            name="ValidatorAgent",
            instructions=VALIDATOR_AGENT_SYSTEM_PROMPT,
            tools=tools,
            model=self.model,
            **(self.model_kwargs or {}),
        )

        prompt = self._build_prompt(
            task=task,
            strong_dir=strong_dir,
            weak_dir=weak_dir,
            strong_agent_id=strong_agent_id,
            weak_agent_id=weak_agent_id,
            strong_produced_files=strong_produced_files,
            weak_produced_files=weak_produced_files,
            golden_answer_file=golden_answer_file,
            golden_answer_kind=golden_answer_kind,
            report_path=report_path,
            feedback_path=feedback_path,
        )

        stream = Runner.run_streamed(agent, input=prompt)
        async for event in stream.stream_events():
            if on_event:
                on_event(event)
        self._stream_results.append(stream)

        # Parse report if agent wrote it; fall back to a minimal error report
        if report_path.exists():
            try:
                return json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not parse validation_report.json: %s", exc)

        # Fallback: write a minimal report so the pipeline doesn't crash
        fallback = {
            "strong_agent": strong_agent_id,
            "weak_agent": weak_agent_id,
            "verdict": "FAIL",
            "result_quality": {
                "files": {"present": [], "missing_from_weak": strong_produced_files, "extra_in_weak": []},
                "scores": {"file_presence_ratio": 0.0, "content_match_ratio": 0.0, "overall_quality_score": 0.0},
            },
            "analysis": "ValidatorAgent did not write a report.",
            "verdict_reasoning": "Fallback FAIL due to missing report.",
        }
        report_path.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        if not feedback_path.exists():
            feedback_path.write_text(
                "Verdict      : FAIL\nFile presence: 0.0\nContent match: 0.0\n"
                "Analysis: ValidatorAgent did not produce output.\n",
                encoding="utf-8",
            )
        return fallback

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        task: str,
        strong_dir: Path,
        weak_dir: Path,
        strong_agent_id: str,
        weak_agent_id: str,
        strong_produced_files: list[str],
        weak_produced_files: list[str] | None,
        report_path: Path,
        feedback_path: Path,
        golden_answer_file: Path | None = None,
        golden_answer_kind: str = "officeqa",
    ) -> str:
        # List what's actually on disk for the agent's awareness
        # Exclude intermediate/execution files (.py, .sh, etc.) — these are never final outputs
        try:
            strong_files = sorted(p.name for p in strong_dir.iterdir() if _is_output_file(p))
        except Exception:
            strong_files = strong_produced_files

        try:
            all_weak = sorted(p.name for p in weak_dir.iterdir() if _is_output_file(p))
            if weak_produced_files:
                weak_set = set(weak_produced_files)
                weak_files = [f for f in all_weak if f in weak_set]
            else:
                weak_files = all_weak
        except Exception:
            weak_files = weak_produced_files or []

        # Use strong_produced_files as authoritative expected list when available
        expected_note = (
            f"Expected output files (authoritative, from strong agent run): {strong_produced_files}"
            if strong_produced_files
            else f"Expected output files (inferred from strong dir, excluding execution artifacts): {strong_files}"
        )
        weak_note = (
            f"Weak agent declared outputs: {weak_produced_files}"
            if weak_produced_files
            else ""
        )

        if golden_answer_file is not None and golden_answer_kind == "dabench":
            return f"""Task:
{task}

Strong agent id  : {strong_agent_id}
Weak agent id    : {weak_agent_id}
Weak output dir  : {weak_dir}
Golden answer    : {golden_answer_file}

This is a CLOSED-FORM MULTI-VALUE task (DABench). The answer format is one or more
`@name[value]` sub-answers written to output.txt (e.g. `@mean_fare[34.65] @std_fare[12.10]`).
The golden answer is a JSON file: a list of [name, value] pairs — NOT a single scalar, so do
NOT use compare_answer_fuzzy (that tool assumes one tolerance-matched number and will
mis-grade multi-value answers).
Use compare_answer_dabench, which parses weak's `@name[value]` lines and checks EACH named
sub-answer independently: correct iff it string-matches the golden value OR
|float(pred) - float(gold)| < 1e-6. The task only PASSES if ALL sub-answers are correct — this
mirrors the official grading exactly, so trust its verdict over your own judgement.
The per-sub-answer `correctness` breakdown it returns is the most useful diagnostic signal:
report in your feedback WHICH named sub-answers were wrong (not just pass/fail), since that is
what downstream gap analysis needs (e.g. one field off by a constant factor suggests a wrong
statistical convention like ddof/skewness-bias; a missing field means output.txt lacked that
`@name[...]` entirely; a value formatted with extra quotes/padding not matching the golden's
literal formatting is a template-compliance error, not a computation error).

Write your results to:
  validation_report.json : {report_path}
  validator_feedback.txt : {feedback_path}

Now execute the validation workflow:
Step 1: compare_answer_dabench("{golden_answer_file}", "{weak_dir}/output.txt")
Step 2: read_file on both files to inspect content
Step 3: write_file to save validation_report.json at {report_path}
Step 4: write_file to save validator_feedback.txt at {feedback_path}

Start with Step 1 now.
"""

        if golden_answer_file is not None:
            return f"""Task:
{task}

Strong agent id  : {strong_agent_id}
Weak agent id    : {weak_agent_id}
Weak output dir  : {weak_dir}
Golden answer    : {golden_answer_file}

This is a TEXT-ANSWER task (e.g. OfficeQA). The expected answer is a single value in
golden_answer.txt; the weak agent writes its answer to output.txt.
Use compare_answer_fuzzy to evaluate the weak agent's output against the golden answer.
This tool runs fuzzy matching at multiple tolerance levels (exact, 0.1%, 1%, 5%).

Write your results to:
  validation_report.json : {report_path}
  validator_feedback.txt : {feedback_path}

Now execute the validation workflow:
Step 1: compare_answer_fuzzy("{golden_answer_file}", "{weak_dir}/output.txt")
Step 2: read_file on both files to inspect content
Step 3: write_file to save validation_report.json at {report_path}
Step 4: write_file to save validator_feedback.txt at {feedback_path}

Start with Step 1 now.
"""

        return f"""Task:
{task}

Strong agent id  : {strong_agent_id}
Weak agent id    : {weak_agent_id}
Strong output dir: {strong_dir}
Weak output dir  : {weak_dir}

{expected_note}
{weak_note}
Files on disk — strong dir (final outputs only): {strong_files}
Files on disk — weak dir   (declared outputs only): {weak_files}

IMPORTANT: Files with extensions .py, .sh, .bash, .pyc, .log, .tmp are execution artifacts,
NOT final outputs. Do NOT count them as expected or missing deliverables.
When strong_produced_files is provided above, use ONLY that list as expected outputs.
When weak agent declared outputs are provided, ignore any other files in the weak directory.

Write your results to:
  validation_report.json : {report_path}
  validator_feedback.txt : {feedback_path}

Now execute the validation workflow:
Step 1: find_content_matches("{strong_dir}", "{weak_dir}")
Step 2: compare_files for each matched pair
Step 3: read_file for unmatched / extra files if needed
Step 4: write_file to save validation_report.json at {report_path}
Step 5: write_file to save validator_feedback.txt at {feedback_path}

Start with Step 1 now.
"""

    # ── tools ─────────────────────────────────────────────────────────────────

    def _build_tools(self, strong_dir: Path, weak_dir: Path, out_dir: Path, expected_files: list[str] | None = None, weak_produced_files: list[str] | None = None, golden_answer_file: Path | None = None, golden_answer_kind: str = "officeqa") -> list:
        project_root = Path.cwd()
        allowed_dirs = [strong_dir.resolve(), weak_dir.resolve(), out_dir.resolve()]

        def _safe_path(path_str: str) -> Path:
            p = Path(path_str)
            if not p.is_absolute():
                p = project_root / path_str
            p = p.resolve()
            # Allow reads under any of the allowed dirs or project root
            allowed = [project_root.resolve()] + allowed_dirs
            if not any(str(p).startswith(str(a)) for a in allowed):
                raise ValueError(f"Path '{path_str}' is outside allowed directories.")
            return p

        _weak_dir_resolved  = weak_dir.resolve()
        _weak_produced_set  = set(weak_produced_files) if weak_produced_files else None

        @function_tool
        def list_files(directory: str) -> str:
            """List files in a directory with their sizes and extensions."""
            try:
                d = Path(directory)
                if not d.is_absolute():
                    d = project_root / directory
                d = d.resolve()
                if not d.is_dir():
                    return f"Error: '{directory}' is not a directory."
                is_weak = str(d).startswith(str(_weak_dir_resolved))
                lines = []
                for p in sorted(d.iterdir()):
                    if not _is_output_file(p):
                        continue
                    if is_weak and _weak_produced_set and p.name not in _weak_produced_set:
                        continue
                    size = p.stat().st_size
                    lines.append(f"{p.name}  ({size} bytes)  [{p.suffix}]")
                return "\n".join(lines) if lines else "(empty directory, or only execution artifacts present)"
            except Exception as exc:
                return f"Error: {exc}"

        @function_tool
        def read_file(file_path: str) -> str:
            """Read a file's content (capped at 8000 chars). Path may be absolute or relative to project root."""
            try:
                p = _safe_path(file_path)
                if not p.exists():
                    return f"Error: file '{file_path}' not found."
                text = p.read_text(encoding="utf-8", errors="replace")
                if len(text) > 8000:
                    text = text[:8000] + f"\n... (truncated, total {len(text)} chars)"
                return text
            except Exception as exc:
                return f"Error: {exc}"

        @function_tool
        def find_content_matches(strong_directory: str, weak_directory: str) -> str:
            """
            Match files from strong_directory to weak_directory by content fingerprint.
            Ignores filenames — only data content determines a match.
            Returns JSON with matched pairs, missing files, and extra files.
            """
            try:
                s_dir = Path(strong_directory)
                w_dir = Path(weak_directory)
                if not s_dir.is_absolute():
                    s_dir = project_root / strong_directory
                if not w_dir.is_absolute():
                    w_dir = project_root / weak_directory

                if not s_dir.is_dir():
                    return json.dumps({"error": f"strong_directory not found: {strong_directory}"})
                if not w_dir.is_dir():
                    return json.dumps({"error": f"weak_directory not found: {weak_directory}"})

                strong_files = {p.name: p for p in s_dir.iterdir() if _is_output_file(p)}
                # Restrict to authoritative expected list when provided
                if expected_files:
                    expected_set = set(expected_files)
                    strong_files = {n: p for n, p in strong_files.items() if n in expected_set}
                weak_files = {p.name: p for p in w_dir.iterdir() if _is_output_file(p)}
                # Restrict weak files to what the weak agent declared it produced
                if weak_produced_files:
                    weak_set = set(weak_produced_files)
                    weak_files = {n: p for n, p in weak_files.items() if n in weak_set}

                pairs = _match_files_by_content(strong_files, dict(weak_files))
                matched_strong = {s for s, _ in pairs}
                matched_weak   = {w for _, w in pairs}

                result = {
                    "matched": [
                        {"strong_file": s, "weak_file": w,
                         "strong_path": str(s_dir / s), "weak_path": str(w_dir / w)}
                        for s, w in pairs
                    ],
                    "missing_from_weak": sorted(n for n in strong_files if n not in matched_strong),
                    "extra_in_weak": sorted(n for n in weak_files if n not in matched_weak),
                    "strong_files": sorted(strong_files),
                    "weak_files": sorted(weak_files),
                    "file_presence_ratio": round(
                        len(pairs) / len(strong_files) if strong_files else 1.0, 4
                    ),
                }
                return json.dumps(result, indent=2)
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        @function_tool
        def compare_files(strong_path: str, weak_path: str) -> str:
            """
            Run type-aware programmatic comparison between a strong reference file
            and a weak output file. Returns JSON with content_match, comparator type,
            and detailed stats (row match ratio for CSV, key diff for JSON, etc.).
            """
            try:
                s = Path(strong_path)
                w = Path(weak_path)
                if not s.is_absolute():
                    s = project_root / strong_path
                if not w.is_absolute():
                    w = project_root / weak_path

                if not s.exists():
                    return json.dumps({"error": f"strong file not found: {strong_path}"})
                if not w.exists():
                    return json.dumps({"error": f"weak file not found: {weak_path}"})

                result = compare_file(s, w, weak_filename=w.name)
                return json.dumps(result.to_dict(), indent=2)
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        @function_tool
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file (must be inside the output directory)."""
            try:
                p = Path(file_path)
                if not p.is_absolute():
                    p = out_dir / file_path
                p = p.resolve()
                if not str(p).startswith(str(out_dir.resolve())):
                    return f"Error: can only write inside output dir ({out_dir})."
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Wrote {len(content)} chars to {p}"
            except Exception as exc:
                return f"Error writing file: {exc}"

        tools = [list_files, read_file, find_content_matches, compare_files, write_file]

        if golden_answer_file is not None and golden_answer_kind == "dabench":
            _golden_answer_file = golden_answer_file

            @function_tool
            def compare_answer_dabench(golden_path: str, weak_output_path: str) -> str:
                """
                Compare a weak agent's multi-value @name[value] answer against the golden
                answer (a JSON file of [name, value] pairs). For each named sub-answer:
                correct iff it string-matches the golden value OR
                |float(pred) - float(gold)| < 1e-6. The task passes iff ALL sub-answers are
                correct — mirrors DABench's official closed-form grading exactly.
                Returns JSON with golden, predicted, per-sub-answer `correctness`, n_correct/
                n_total, and verdict (PASS/PARTIAL/FAIL). Use this for DABench tasks instead
                of compare_files or compare_answer_fuzzy (which assumes a single scalar).
                """
                import re as _re
                try:
                    gp = Path(golden_path) if Path(golden_path).is_absolute() else project_root / golden_path
                    wp = Path(weak_output_path) if Path(weak_output_path).is_absolute() else project_root / weak_output_path

                    if not gp.exists():
                        return json.dumps({"error": f"Golden answer not found: {golden_path}"})
                    if not wp.exists():
                        return json.dumps({"error": f"Weak output not found: {weak_output_path}"})

                    golden_pairs = json.loads(gp.read_text(encoding="utf-8"))
                    label = {str(name): str(val) for name, val in golden_pairs}

                    text = wp.read_text(encoding="utf-8")
                    pred = dict(_re.findall(r"@(\w+)\[(.*?)\]", text))

                    def _is_equal(r, l: str) -> bool:
                        if r is None:
                            return False
                        if str(r) == l:
                            return True
                        try:
                            return abs(float(r) - float(l)) < 1e-6
                        except Exception:
                            return False

                    correctness = {k: _is_equal(pred.get(k), v) for k, v in label.items()}
                    n_ok = sum(1 for ok in correctness.values() if ok)
                    n_total = len(correctness)
                    passed = n_total > 0 and n_ok == n_total
                    verdict = "PASS" if passed else ("PARTIAL" if n_ok > 0 else "FAIL")

                    return json.dumps({
                        "golden":       label,
                        "predicted":    pred,
                        "correctness":  correctness,
                        "n_correct":    n_ok,
                        "n_total":      n_total,
                        "verdict":      verdict,
                        "overall_quality_score": (n_ok / n_total) if n_total else 0.0,
                    }, indent=2)
                except Exception as exc:
                    return json.dumps({"error": str(exc)})

            tools.append(compare_answer_dabench)

        elif golden_answer_file is not None:
            _golden_answer_file = golden_answer_file

            @function_tool
            def compare_answer_fuzzy(golden_path: str, weak_output_path: str) -> str:
                """
                Compare a weak agent's text answer against the golden answer using
                fuzzy matching. Evaluates at multiple tolerance levels:
                  exact (0%), 0.1%, 1%, 5%.
                Returns JSON with scores_by_tolerance, verdict, and rationale.
                Use this tool for text-answer tasks (e.g. OfficeQA) instead of compare_files.
                """
                import os as _os
                import sys as _sys
                # reward.py lives in <project_root>/officeqa/
                reward_dir = _os.environ.get("OFFICEQA_DIR") or str(project_root.parent / "officeqa")
                try:
                    if reward_dir not in _sys.path:
                        _sys.path.insert(0, reward_dir)
                    from reward import fuzzy_match_answer, extract_final_answer  # type: ignore
                except Exception as exc:
                    return json.dumps({"error": f"Could not import reward.py: {exc}"})

                try:
                    gp = Path(golden_path) if Path(golden_path).is_absolute() else project_root / golden_path
                    wp = Path(weak_output_path) if Path(weak_output_path).is_absolute() else project_root / weak_output_path

                    if not gp.exists():
                        return json.dumps({"error": f"Golden answer not found: {golden_path}"})
                    if not wp.exists():
                        return json.dumps({"error": f"Weak output not found: {weak_output_path}"})

                    ground_truth = gp.read_text(encoding="utf-8").strip()
                    raw_output   = wp.read_text(encoding="utf-8").strip()
                    predicted    = extract_final_answer(raw_output)

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
                    score_5pct = scores_by_tolerance["5pct"]
                    verdict = "PASS" if exact_pass else ("PARTIAL" if score_5pct > 0 else "FAIL")

                    return json.dumps({
                        "golden":              ground_truth,
                        "predicted":           predicted,
                        "scores_by_tolerance": scores_by_tolerance,
                        "rationales":          rationales,
                        "verdict":             verdict,
                        "overall_quality_score": score_5pct,
                    }, indent=2)
                except Exception as exc:
                    return json.dumps({"error": str(exc)})

            tools.append(compare_answer_fuzzy)

        return tools
