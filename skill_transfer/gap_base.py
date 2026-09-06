"""Shared primitives for the Diagnoser and the Patcher.

Holds the execution-structure distance (graded GED over tier-1/tier-2 labels),
the per-case context object, the local file editor the agents use to read and
rewrite a skill, and the tool factories both agents build on.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Runner, FunctionTool, function_tool
from agents.memory.sqlite_session import SQLiteSession
from agents.tool_context import ToolContext
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.apply_diff import apply_diff

from execution_structure import ExecutionStructure

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CaseContext — per-case paths for multi-data mode
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseContext:
    """Paths for one (strong, weak) training case in multi-data mode."""
    case_id: str
    strong_structure_path: str
    weak_structure_path: str
    strong_traj_path: str
    weak_traj_path: str


# ──────────────────────────────────────────────────────────────────────────────
# Structural distance
# ──────────────────────────────────────────────────────────────────────────────

def _node_sub_cost(a_t1: str, a_t2: str, b_t1: str, b_t2: str) -> float:
    """
    Substitution cost between two nodes labelled by (tier1, tier2).

    | tier1 match | tier2 match | cost |
    |-------------|-------------|------|
    | yes         | yes         | 0.0  |
    | yes         | no          | 0.5  |
    | no          | yes         | 0.75 |
    | no          | no          | 1.0  |
    """
    t1_match = a_t1 == b_t1
    t2_match = a_t2 == b_t2
    if t1_match and t2_match:
        return 0.0
    if t1_match:
        return 0.5
    if t2_match:
        return 0.75
    return 1.0


def graph_edit_distance(
    strong: ExecutionStructure,
    weak: ExecutionStructure,
    *,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """Graph Edit Distance between two execution structures modelled as path-graphs."""
    s_nodes = strong.nodes
    w_nodes = weak.nodes
    m, n = len(s_nodes), len(w_nodes)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + delete_cost
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insert_cost

    for i in range(1, m + 1):
        s = s_nodes[i - 1]
        for j in range(1, n + 1):
            w = w_nodes[j - 1]
            sub    = dp[i - 1][j - 1] + _node_sub_cost(
                s.category_tier1, s.category_tier2,
                w.category_tier1, w.category_tier2,
            )
            delete = dp[i - 1][j] + delete_cost
            insert = dp[i][j - 1] + insert_cost
            dp[i][j] = min(sub, delete, insert)

    return dp[m][n]


def normalised_ged(strong: ExecutionStructure, weak: ExecutionStructure) -> float:
    """GED normalised to [0, 1].  0.0 = identical, 1.0 = maximally different."""
    raw   = graph_edit_distance(strong, weak)
    worst = float(len(strong.nodes) + len(weak.nodes))
    if worst == 0.0:
        return 0.0
    return min(raw / worst, 1.0)


@dataclass
class StructuralDistance:
    """Structural comparison between a strong and weak execution structure."""

    raw_ged: float
    normalised_ged: float

    tier1_only_strong: set[str]
    tier1_only_weak: set[str]
    tier1_shared: set[str]

    tier2_only_strong: set[str]
    tier2_only_weak: set[str]
    tier2_shared: set[str]

    strong_node_count: int
    weak_node_count: int

    @property
    def similarity(self) -> float:
        """1 - normalised_ged."""
        return 1.0 - self.normalised_ged

    def to_dict(self) -> dict:
        return {
            "raw_ged": round(self.raw_ged, 4),
            "normalised_ged": round(self.normalised_ged, 4),
            "similarity": round(self.similarity, 4),
            "tier1": {
                "only_strong": sorted(self.tier1_only_strong),
                "only_weak":   sorted(self.tier1_only_weak),
                "shared":      sorted(self.tier1_shared),
            },
            "tier2": {
                "only_strong": sorted(self.tier2_only_strong),
                "only_weak":   sorted(self.tier2_only_weak),
                "shared":      sorted(self.tier2_shared),
            },
            "node_counts": {
                "strong": self.strong_node_count,
                "weak":   self.weak_node_count,
            },
        }


def compute_structural_distance(
    strong: ExecutionStructure,
    weak: ExecutionStructure,
) -> StructuralDistance:
    """Compute and return the full StructuralDistance between two structures."""
    raw  = graph_edit_distance(strong, weak)
    norm = normalised_ged(strong, weak)
    s_t1 = strong.tier1_categories()
    w_t1 = weak.tier1_categories()
    s_t2 = strong.tier2_labels()
    w_t2 = weak.tier2_labels()
    return StructuralDistance(
        raw_ged=raw,
        normalised_ged=norm,
        tier1_only_strong=s_t1 - w_t1,
        tier1_only_weak=w_t1 - s_t1,
        tier1_shared=s_t1 & w_t1,
        tier2_only_strong=s_t2 - w_t2,
        tier2_only_weak=w_t2 - s_t2,
        tier2_shared=s_t2 & w_t2,
        strong_node_count=len(strong.nodes),
        weak_node_count=len(weak.nodes),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Score-weighting constants
# ──────────────────────────────────────────────────────────────────────────────

FAIL_WEIGHT: float = 1.0
PASS_WEIGHT: float = 0.3
SCORE_THRESHOLD: float = 0.2

# gap_filter_mode: how submit_gap_report decides which gaps enter recommended_patch_order.
#   "score"  — legacy: score = (fw·|fail| + pw·|pass|)/n_cases, admit if score ≥ threshold
#              AND count ≥ min_gap_count.  Threshold depends on n_cases.
#   "binary" — explicit rule: admit if |failed_case_ids| ≥ 1  OR  |pass_case_ids| ≥ 2.
#              No score math; the score field is still written for diagnostic display
#              but ignored for filtering.  Designed to be intuitive and case-count
#              agnostic.  RESOLVED gaps still excluded; min_gap_count still applied.
GAP_FILTER_MODE: str = "score"
BINARY_MIN_FAILED: int = 1
BINARY_MIN_PASSED: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# Backup path helper
# ──────────────────────────────────────────────────────────────────────────────

def _skill_backup_path(file_path: Path, project_root: Path) -> Path:
    """Compute backup path under skills_backup/, mirroring skill structure."""
    import time
    backup_root = project_root.resolve() / "skills_backup"
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        rel = Path(file_path.name)
    parts = rel.parts
    if parts and parts[0] == "skills":
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(file_path.name)
    bak_name = rel.name + f".bak_{int(time.time())}"
    return backup_root / rel.parent / bak_name


# ──────────────────────────────────────────────────────────────────────────────
# LocalFileEditor
# ──────────────────────────────────────────────────────────────────────────────

class LocalFileEditor:
    def __init__(self, project_root: Path, diff_callback=None):
        self.project_root = project_root.resolve()
        self.diff_callback = diff_callback

    def _validate_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.resolve()
        if not str(p).startswith(str(self.project_root)):
            raise ValueError(f"Path '{path_str}' escapes the project directory.")
        return p

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if path.exists():
            return ApplyPatchResult(status="failed", output=f"File already exists: {operation.path}")
        try:
            content = apply_diff("", operation.diff or "", mode="create")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ApplyPatchResult(status="completed", output=f"Created {operation.path} ({len(content)} chars)")
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(status="failed", output=f"File not found: {operation.path}")
        try:
            original = path.read_text(encoding="utf-8")
            backup_path = _skill_backup_path(path, self.project_root)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_text(original, encoding="utf-8")

            diff_text = operation.diff or ""
            lines = diff_text.split("\n")
            first_anchor = next((i for i, l in enumerate(lines) if l.startswith("@@")), -1)
            if first_anchor != -1:
                diff_text = "\n".join(lines[first_anchor:])
            else:
                diff_text = "\n".join(l for l in lines if not (l.startswith("--- ") or l.startswith("+++ ")))

            updated = apply_diff(original, diff_text)
            path.write_text(updated, encoding="utf-8")
            if self.diff_callback:
                self.diff_callback(str(operation.path), original, updated)
            return ApplyPatchResult(
                status="completed",
                output=f"Updated {operation.path} (backup: {backup_path})",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(status="failed", output=f"File not found: {operation.path}")
        try:
            path.unlink()
            return ApplyPatchResult(status="completed", output=f"Deleted {operation.path}")
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Skill directory expansion helper
# ──────────────────────────────────────────────────────────────────────────────

def _file_role(f: Path) -> str:
    """Return a short patchability hint for any skill file."""
    ext = f.suffix.lower()
    if ext in (".py", ".sh", ".js", ".ts"):
        return "executable script — patch here if the script logic is the root of failure"
    return "skill file — patch here if its content is the root of failure"


def _expand_skill_listing(skill_paths: list[str]) -> str:
    """Build a rich skill listing showing all patchable files per skill directory."""
    lines: list[str] = []
    for sp in skill_paths:
        skill_dir = Path(sp).parent
        skill_name = skill_dir.name
        lines.append(f"  Skill: {skill_name}  ({skill_dir}/)")
        try:
            for f in sorted(skill_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(skill_dir)
                role = _file_role(f)
                lines.append(f"    {rel}  — {role}  :  {f}")
        except Exception:
            lines.append(f"    (could not list files in {skill_dir})")
        lines.append("")
    return "\n".join(lines).rstrip()


# ──────────────────────────────────────────────────────────────────────────────
# GapAgentBase
#
# Foundation for any agent in the gap-analysis pipeline.  Provides the
# streaming entry point, cost/session machinery, and tool factory methods
# but does NOT impose any system prompt — subclasses supply their own.
# ──────────────────────────────────────────────────────────────────────────────

class GapAgentBase:
    """Base class for split-framework agents (GapDiagnoser, SkillPatcher).

    Subclasses must provide:
      • system_prompt_multi   — built by their own _build_*_system_prompt()
      • _build_tools()        — returns base tools (read_file, write_file, etc.)
      • _build_multi_tools()  — returns multi-case tools the agent actually uses
      • _build_prompt_multi() — builds the per-run input prompt
    """

    # Identification — subclasses should override
    _AGENT_NAME: str = "GapAgentBase"

    # Standard resume cues (subclasses may override for finer-grained recovery)
    _RESUME_INPUT = "Continue from the previous step and complete the task."
    _TEXT_ONLY_RESUME_INPUT = (
        "You stopped early and did not complete the task. "
        "Continue from where you left off using TOOL CALLS, not prose. "
        "DO NOT early stop or produce another text-only response."
    )

    def __init__(
        self,
        model: str = "gpt-5.4",
        system_prompt_multi: str | None = None,
        model_kwargs: dict | None = None,
        project_root: str | Path | None = None,
        task_dir: str | Path | None = None,
        diff_log_path: str | Path | None = None,
        trajectory_format: str = "md",
        fail_weight: float = FAIL_WEIGHT,
        pass_weight: float = PASS_WEIGHT,
        score_threshold: float = SCORE_THRESHOLD,
        min_gap_count: int = 2,
        gap_filter_mode: str = GAP_FILTER_MODE,
    ):
        if system_prompt_multi is None:
            raise ValueError(
                f"{type(self).__name__}: system_prompt_multi is required.  "
                "Subclasses must build their own prompt via _build_*_system_prompt()."
            )
        self.model = model
        self.fail_weight = fail_weight
        self.pass_weight = pass_weight
        self.score_threshold = score_threshold
        self.min_gap_count = min_gap_count
        if gap_filter_mode not in ("score", "binary"):
            raise ValueError(f"gap_filter_mode must be 'score' or 'binary', got {gap_filter_mode!r}")
        self.gap_filter_mode = gap_filter_mode
        self.system_prompt_multi = system_prompt_multi
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.task_dir = Path(task_dir) if task_dir else self.project_root
        self.diff_log_path = Path(diff_log_path) if diff_log_path else None
        self.trajectory_format = trajectory_format if trajectory_format in ("md", "json") else "md"
        self.last_thought: str | None = None
        self._skill_dirs: list[Path] = []  # set per-run; diff only recorded for files inside these dirs
        self._model_kwargs = model_kwargs or {}
        self.tools = self._build_tools()
        # Note: each agent's working `Agent` instance is built fresh inside
        # run_streamed_multi() so multi-case tools + multi-prompt are wired
        # in one place.  We do NOT pre-build an agent here.

    # ── public streaming entry point ────────────────────────────────────────

    def run_streamed_multi(
        self,
        task_description: str,
        cases: list[CaseContext],
        skill_paths: list[str],
        out_dir: str,
        force_read_trajectories: bool = False,
        prev_gap_trajectory_paths: list[str] | None = None,
        prev_gap_patch_narrative_paths: list[str] | None = None,
        instruction_save_path: str | None = None,
        case_history_path: str | None = None,
        max_turns: int = 50,
        session: SQLiteSession | None = None,
        resuming: bool = False,
        resume_input: str | None = None,
        **kwargs: Any,
    ):
        """Run multi-case analysis with the subclass's tool set + prompt.

        Builds a fresh `Agent` per call so multi-case readers can capture the
        current `cases` list in their closures.  Extra kwargs are forwarded
        to the subclass's `_build_prompt_multi()` (useful for e.g. the
        Patcher's `gap_report_path` argument).
        """
        multi_tools = self.tools + self._build_multi_tools(
            cases=cases,
            case_history_path=case_history_path,
            out_dir=out_dir,
            prev_gap_narrative_paths=prev_gap_patch_narrative_paths,
        )
        multi_agent = Agent(
            name=self._AGENT_NAME,
            instructions=self.system_prompt_multi,
            tools=multi_tools,
            model=self.model,
            **self._model_kwargs,
        )

        if resuming and session is not None:
            input_text = resume_input if resume_input is not None else self._RESUME_INPUT
        else:
            input_text = self._build_prompt_multi(
                task_description=task_description,
                cases=cases,
                skill_paths=skill_paths,
                out_dir=out_dir,
                force_read_trajectories=force_read_trajectories,
                prev_gap_trajectory_paths=prev_gap_trajectory_paths,
                prev_gap_patch_narrative_paths=prev_gap_patch_narrative_paths,
                **kwargs,
            )

        if instruction_save_path and not resuming:
            try:
                Path(instruction_save_path).write_text(input_text, encoding="utf-8")
            except Exception as exc:
                logger.error(f"Failed to save instruction to {instruction_save_path}: {exc}")

        self._skill_dirs = [Path(sp).parent.resolve() for sp in skill_paths]
        return Runner.run_streamed(multi_agent, input=input_text, max_turns=max_turns,
                                   session=session)

    # ── subclass hooks (must override) ───────────────────────────────────────

    def _build_prompt_multi(self, **kwargs: Any) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_prompt_multi()"
        )

    def _build_tools(self) -> list:
        """Default base tool set: read_file + write_file only.  Subclasses
        override to add patch tools / activate_skill / etc."""
        return [self._make_read_file_tool(), self._make_write_file_tool()]

    def _build_multi_tools(
        self,
        cases: list[CaseContext],
        case_history_path: str | None = None,
        out_dir: str | None = None,
        prev_gap_narrative_paths: list[str] | None = None,
    ) -> list:
        """Default: full multi-case tool set.

        Subclasses override to filter down to the subset they actually need
        (e.g. Diagnoser drops finalize_patches; Patcher drops the diagnosis
        tools but keeps fallback case-readers).
        """
        case_map = {c.case_id: c for c in cases}
        root = self.project_root
        _hist_path = case_history_path
        self_ref = self  # for closures

        # ── read_case_history ─────────────────────────────────────────────────
        @function_tool
        def read_case_history() -> str:
            """Read the iteration history for all training cases: pass/fail, GED (structural
            distance from strong agent), cell_match_ratio (tabular tasks), and
            scores_by_tolerance (text-answer tasks, e.g. OfficeQA) for each completed iteration.
            Call this as the very first action before any other step."""
            if not _hist_path or not Path(_hist_path).exists():
                return (
                    "First iteration (iter 0) — no previous history available. "
                    "Proceed directly to Step 1 (list_cases)."
                )
            try:
                history = json.loads(Path(_hist_path).read_text(encoding="utf-8"))
            except Exception as exc:
                return f"Error reading case history: {exc}"
            updated_at = history.get("updated_at_iter", 0)
            if updated_at == 0 and all(
                len(v.get("iter_history", [])) <= 1
                for v in history.get("cases", {}).values()
            ):
                return (
                    "ITER 0 BASELINE — only one iteration of data exists. "
                    "No cross-iteration trend or regression analysis is possible yet. "
                    "Proceed immediately to list_cases() to see current pass/fail status "
                    "and run the full gap analysis on all cases (failing and passing).\n\n"
                    + json.dumps(history, indent=2, ensure_ascii=False)
                )
            return json.dumps(history, indent=2, ensure_ascii=False)

        @function_tool
        def list_cases() -> str:
            """List all cases in the pool with their current pass/fail status,
            tier1/tier2 gap signatures, and normalised structural distance."""
            current_pass: dict[str, bool | None] = {}
            if _hist_path and Path(_hist_path).exists():
                try:
                    hist = json.loads(Path(_hist_path).read_text(encoding="utf-8"))
                    for cid, cdata in hist.get("cases", {}).items():
                        entries = cdata.get("iter_history", [])
                        if entries:
                            latest = max(entries, key=lambda e: e["iter"])
                            current_pass[cid] = latest.get("pass")
                except Exception:
                    pass

            result = []
            failing_ids: list[str] = []
            passing_ids: list[str] = []
            unknown_ids: list[str] = []
            for c in cases:
                pass_val = current_pass.get(c.case_id)
                if pass_val is True:
                    status = "PASSING"
                    passing_ids.append(c.case_id)
                elif pass_val is False:
                    status = "FAILING"
                    failing_ids.append(c.case_id)
                else:
                    status = "UNKNOWN"
                    unknown_ids.append(c.case_id)
                entry: dict = {"case_id": c.case_id, "current_status": status}
                try:
                    s = ExecutionStructure.model_validate(
                        json.loads(Path(c.strong_structure_path).read_text(encoding="utf-8"))
                    )
                    w = ExecutionStructure.model_validate(
                        json.loads(Path(c.weak_structure_path).read_text(encoding="utf-8"))
                    )
                    dist = compute_structural_distance(s, w)
                    entry["tier1_only_strong"] = sorted(dist.tier1_only_strong)
                    entry["tier2_only_strong"] = sorted(dist.tier2_only_strong)
                    entry["normalised_ged"]    = round(dist.normalised_ged, 3)
                    entry["strong_steps"]      = dist.strong_node_count
                    entry["weak_steps"]        = dist.weak_node_count
                except Exception as exc:
                    entry["error"] = str(exc)
                result.append(entry)
            output = {
                "cases": result,
                "summary": {
                    "failing_case_ids": failing_ids,
                    "passing_case_ids": passing_ids,
                    **({"unknown_case_ids": unknown_ids} if unknown_ids else {}),
                    "note": (
                        "Gap analysis covers ALL cases. "
                        f"FAILING cases are primary evidence (weight {self_ref.fail_weight}). "
                        f"PASSING cases contribute robustness-gap evidence (weight {self_ref.pass_weight}) "
                        "and are also regression guards — patches must not break them."
                    ),
                },
            }
            return json.dumps(output, indent=2)

        @function_tool
        def read_case_structures(case_id: str) -> str:
            """Read both the strong and weak ExecutionStructure JSONs for one case.
            Call for every case during the deep-dive phase."""
            c = case_map.get(case_id)
            if not c:
                return f"Error: case_id '{case_id}' not found."
            try:
                strong_text = Path(c.strong_structure_path).read_text(encoding="utf-8")
                weak_text   = Path(c.weak_structure_path).read_text(encoding="utf-8")
                payload: dict = {
                    "case_id":          case_id,
                    "strong_structure": json.loads(strong_text),
                    "weak_structure":   json.loads(weak_text),
                }
                return json.dumps(payload, indent=2)
            except Exception as exc:
                return f"Error reading structures for '{case_id}': {exc}"

        _traj_fmt = self.trajectory_format

        @function_tool
        def read_case_trajectory(case_id: str, agent: str) -> str:
            """Read a trajectory file for one case.
            agent must be 'strong' or 'weak'.
            Use only when code_snippets in the structures are insufficient
            to understand a specific gap — trajectories are large."""
            c = case_map.get(case_id)
            if not c:
                return f"Error: case_id '{case_id}' not found."
            if agent not in ("strong", "weak"):
                return "Error: agent must be 'strong' or 'weak'."
            base_path = Path(c.strong_traj_path if agent == "strong" else c.weak_traj_path)
            if _traj_fmt == "json":
                path = base_path.with_suffix(".json")
            else:
                path = base_path
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                alt = base_path.with_suffix(".md" if _traj_fmt == "json" else ".json")
                if not alt.is_absolute():
                    alt = root / alt
                if alt.exists():
                    path = alt
                else:
                    return f"Error: trajectory not found: {path}"
            try:
                return path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Error reading trajectory: {exc}"

        @function_tool
        def read_case_validator_report(case_id: str) -> str:
            """Read the full validation report for one case from the current iteration."""
            c = case_map.get(case_id)
            if not c:
                return f"Error: case_id '{case_id}' not found."
            validator_dir = Path(c.weak_structure_path).parent / "validator"
            report_path   = validator_dir / "validation_report.json"
            if not report_path.exists():
                return (
                    f"[No validator report found for case {case_id} — "
                    "validation may not have run yet for this iteration]"
                )
            try:
                task_type = "spreadsheetbench"
                meta_path = Path(c.weak_structure_path).parent.parent.parent / "task_meta.json"
                if meta_path.exists():
                    try:
                        task_type = json.loads(meta_path.read_text()).get("type", "spreadsheetbench")
                    except Exception:
                        pass
                if task_type == "officeqa":
                    field_guide = (
                        "# Field guide (officeqa): verdict=PASS/PARTIAL/FAIL | "
                        "scores_by_tolerance: exact=strict, 0.1pct/1pct/5pct=numeric tolerance | "
                        "overall_quality=score at 5pct\n"
                    )
                else:
                    field_guide = (
                        "# Field guide (xlsx): verdict=PASS/PARTIAL/FAIL | "
                        "content_match=BINARY (1.0 only if byte-identical, not a cell measure) | "
                        "cell_match_ratio=continuous cell match | "
                        "per_column_match_ratio=which columns differ\n"
                    )
                report_text = report_path.read_text(encoding="utf-8").strip()
                return f"=== Validator Report: {case_id} ===\n{field_guide}\n{report_text}"
            except Exception as exc:
                return f"Error reading validator report for '{case_id}': {exc}"

        @function_tool
        def read_prev_case_structures(case_id: str) -> str:
            """Read the previous iteration's weak ExecutionStructure for one case."""
            c = case_map.get(case_id)
            if not c:
                return f"Error: case_id '{case_id}' not found."
            current_weak = Path(c.weak_structure_path)
            iter_dir = current_weak.parent
            iter_name = iter_dir.name
            try:
                iter_num = int(iter_name.replace("iter_", ""))
            except ValueError:
                return f"Error: cannot parse iteration number from '{iter_dir}'."
            if iter_num == 0:
                return (
                    f"No previous iteration for case {case_id} — this is iter_00. "
                    "STATUS ANALYSIS is not applicable."
                )
            prev_iter = f"iter_{iter_num - 1:02d}"
            prev_path = iter_dir.parent / prev_iter / "weak_structure.json"
            if not prev_path.exists():
                return f"Previous weak structure not found: {prev_path}"
            try:
                return json.dumps({
                    "case_id": case_id,
                    "prev_iter": prev_iter,
                    "prev_weak_structure": json.loads(prev_path.read_text(encoding="utf-8")),
                }, indent=2)
            except Exception as exc:
                return f"Error reading previous structure for '{case_id}': {exc}"

        # ── submit_gap_report ─────────────────────────────────────────────────
        _gap_report_path = str(Path(out_dir) / "gap_report.json") if out_dir else None
        _n_cases = len(cases)
        _fw = self.fail_weight
        _pw = self.pass_weight
        _st = self.score_threshold
        _filter_mode = self.gap_filter_mode

        @function_tool
        def submit_gap_report(report_file: str = "", report_json: str = "") -> str:
            """Compute gap scores and write gap_report.json.

            Preferred usage — pass report_file (a path written by write_file):
              submit_gap_report(report_file="<out_dir>/gap_report_draft.json")
            Fallback — pass report_json inline (risks token truncation for large reports):
              submit_gap_report(report_json="{...}")

            The report JSON must NOT include 'score' or 'recommended_patch_order' —
            those are computed automatically from failed_case_ids / pass_case_ids.
            """
            if not _gap_report_path:
                return "Error: out_dir not set; cannot write gap_report.json."
            if report_file:
                try:
                    raw = Path(report_file).read_text(encoding="utf-8")
                except Exception as exc:
                    return f"Error reading report_file '{report_file}': {exc}"
                try:
                    report = json.loads(raw)
                except json.JSONDecodeError as exc:
                    return f"Error: report_file is not valid JSON: {exc}"
            elif report_json:
                try:
                    report = json.loads(report_json)
                except json.JSONDecodeError as exc:
                    return f"Error: report_json is not valid JSON: {exc}"
            else:
                return "Error: provide either report_file or report_json."

            _status_order = {"REGRESSION": 0, "PERSISTS": 1, "NEW": 2, "IMPROVEMENT": 3}
            # Always compute the diagnostic score so downstream tools / human inspectors
            # can see the magnitude; whether it drives filtering depends on _filter_mode.
            for gap in report.get("gaps", []):
                n_failed = len(gap.get("failed_case_ids", []))
                n_passed = len(gap.get("pass_case_ids", []))
                score = (_fw * n_failed + _pw * n_passed) / _n_cases if _n_cases > 0 else 0.0
                gap["score"] = round(score, 4)
            _min_count = self.min_gap_count

            def _eligible(gap):
                n_failed = len(gap.get("failed_case_ids", []))
                n_passed = len(gap.get("pass_case_ids", []))
                if gap.get("status", "NEW") == "RESOLVED":
                    return False
                if n_failed + n_passed < _min_count:
                    return False
                if _filter_mode == "binary":
                    return n_failed >= BINARY_MIN_FAILED or n_passed >= BINARY_MIN_PASSED
                # score mode (legacy)
                return gap["score"] >= _st

            eligible = [gap for gap in report.get("gaps", []) if _eligible(gap)]
            # Sort within-tier by (failed desc, then score desc).  In binary mode the score
            # tie-break is still meaningful (more total evidence = earlier patch).
            eligible.sort(key=lambda g: (
                _status_order.get(g.get("status", "NEW"), 2),
                -len(g.get("failed_case_ids", [])),
                -g["score"],
            ))
            report["recommended_patch_order"] = [g["gap_id"] for g in eligible]
            report["fail_weight"] = _fw
            report["pass_weight"] = _pw
            report["score_threshold"] = _st
            report["gap_filter_mode"] = _filter_mode
            report["total_cases"] = _n_cases
            out_path = Path(_gap_report_path)
            pre_decompose_note = ""
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    pre_path = out_path.parent / "gap_report_pre_decompose.json"
                    pre_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
                    pre_decompose_note = f" (previous version backed up to {pre_path.name})"
                out_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                return f"Error writing gap_report.json: {exc}"
            return json.dumps({
                "status": "ok",
                "note": f"gap_report.json written{pre_decompose_note}",
                "written_to": str(out_path),
                "total_gaps": len(report.get("gaps", [])),
                "recommended_patch_order": report["recommended_patch_order"],
                "gap_scores": {g["gap_id"]: g["score"] for g in report.get("gaps", [])},
            }, indent=2)

        # ── finalize_patches ──────────────────────────────────────────────────
        _gap_patches_complete_path = str(Path(out_dir) / "gap_patches_complete") if out_dir else None

        @function_tool
        def finalize_patches() -> str:
            """Signal that all gap patches in recommended_patch_order have been applied.
            Call this as your VERY LAST action after completing the final gap."""
            if not _gap_patches_complete_path:
                return "Error: out_dir not set; cannot write gap_patches_complete."
            try:
                Path(_gap_patches_complete_path).touch()
                return "gap_patches_complete written. All gap patches finalized."
            except Exception as exc:
                return f"Error writing gap_patches_complete: {exc}"

        _prev_gap_narrative_paths = list(prev_gap_narrative_paths or [])
        return [read_case_history, list_cases, read_case_structures,
                read_prev_case_structures, read_case_trajectory,
                read_case_validator_report, submit_gap_report, finalize_patches]

    # ── adaptive reasoning-effort downgrade  (added 2026-05-18) ──────────────

    _REASONING_LADDER = {"high": "medium", "medium": "low", "low": None}

    def _downgrade_reasoning_effort(self) -> str | None:
        """Lower reasoning effort one rung (high→medium→low→none).

        Mutates `self._model_kwargs` in place so the next Agent rebuild picks
        up the new effort.  Returns the new effort name (e.g. "low") or the
        string "none" when reasoning is fully disabled.  Returns None when
        further downgrade is not possible (already at none, or no reasoning
        was configured in the first place).
        """
        try:
            from agents import ModelSettings  # noqa: PLC0415
        except Exception:
            return None
        ms = self._model_kwargs.get("model_settings") if self._model_kwargs else None
        if ms is None:
            return None
        reasoning = getattr(ms, "reasoning", None)
        if reasoning is None:
            return None  # already disabled
        # Extract current effort from either dict or dataclass form
        if isinstance(reasoning, dict):
            current = reasoning.get("effort")
        else:
            current = getattr(reasoning, "effort", None)
        if current not in self._REASONING_LADDER:
            return None
        new = self._REASONING_LADDER[current]
        new_max_tokens = getattr(ms, "max_tokens", None)
        if new is None:
            # Fully disable reasoning
            self._model_kwargs["model_settings"] = ModelSettings(
                reasoning=None,
                max_tokens=new_max_tokens,
            )
            return "none"
        self._model_kwargs["model_settings"] = ModelSettings(
            reasoning={"effort": new},
            max_tokens=new_max_tokens,
        )
        return new

    # ── internal helpers ────────────────────────

    def _list_output_files(self) -> list[str]:
        """List non-trajectory, non-JSON files in task_dir for agent awareness."""
        candidates = []
        try:
            for p in sorted(self.task_dir.rglob("*")):
                if p.is_file() and p.suffix in (".csv", ".txt", ".json", ".md", ".html", ".xlsx", ".xls"):
                    rel = str(p.relative_to(self.project_root))
                    candidates.append(rel)
        except Exception:
            pass
        return candidates[:30]

    def _append_diff(self, file_path: str, old: str, new: str) -> None:
        """Append a unified diff entry to self.diff_log_path (skill files only)."""
        if self.diff_log_path is None:
            return
        if self._skill_dirs:
            resolved = Path(file_path) if Path(file_path).is_absolute() else (self.project_root / file_path)
            resolved = resolved.resolve()
            if not any(resolved.is_relative_to(d) for d in self._skill_dirs):
                return
        rel = file_path
        lines = list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        ))
        if not lines:
            return
        self.diff_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.diff_log_path.open("a", encoding="utf-8") as f:
            thought = getattr(self, "last_thought", None)
            if thought:
                for marker in ("PATCH DRAFT", "```diff", "--- OLD ---"):
                    idx = thought.find(marker)
                    if idx != -1:
                        thought = thought[:idx].rstrip()
                        break
                if thought:
                    f.write(f"### thought\n{thought}\n\n")
            f.write("".join(lines))
            f.write("\n")

    # ── tool factory methods ───────

    def _make_read_file_tool(self):
        root = self.project_root

        @function_tool
        def read_file(file_path: str) -> str:
            """Read a text file.  Path may be absolute or relative to project root.
            Restricted to the project directory."""
            p = Path(file_path)
            if not p.is_absolute():
                p = root / file_path
            p = p.resolve()
            if not str(p).startswith(str(root.resolve())):
                return f"Error: path '{file_path}' is outside the project directory."
            if not p.exists():
                return f"Error: file '{file_path}' not found."
            try:
                return p.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Error reading file: {exc}"

        return read_file

    def _make_write_file_tool(self):
        root = self.project_root
        self_ref = self

        @function_tool
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file, creating parent directories as needed.
            Automatically backs up existing files.
            Do NOT use this tool to edit submitted gap_report.json or
            gap_report_pre_decompose.json; for decomposition, write the updated
            JSON to gap_report_draft.json and call submit_gap_report(report_file=...)."""
            p = Path(file_path)
            if not p.is_absolute():
                p = root / file_path
            p = p.resolve()
            if not str(p).startswith(str(root.resolve())):
                return f"Error: path '{file_path}' is outside the project directory."
            try:
                msg = ""
                if p.exists():
                    original = p.read_text(encoding="utf-8")
                    bak = _skill_backup_path(p, root)
                    bak.parent.mkdir(parents=True, exist_ok=True)
                    bak.write_text(original, encoding="utf-8")
                    self_ref._append_diff(file_path, original, content)
                    msg = f" (original backed up to {bak})"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Wrote {len(content)} chars to {file_path}{msg}"
            except Exception as exc:
                return f"Error writing file: {exc}"

        return write_file

    def _make_replace_in_file_tool(self, out_dir: str | None = None):
        root = self.project_root
        self_ref = self
        out_dir_resolved = Path(out_dir).resolve() if out_dir else None

        async def replace_in_file_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            try:
                args = json.loads(args_json)
                file_path = args["path"]
                old_str   = args["old_str"]
                new_str   = args["new_str"]
            except (KeyError, json.JSONDecodeError) as exc:
                return f"replace_in_file failed: bad arguments — {exc}"
            try:
                p = Path(file_path)
                if not p.is_absolute():
                    p = root / file_path
                p = p.resolve()
                if not str(p).startswith(str(root.resolve())):
                    return f"replace_in_file failed: path escapes project directory."
                protected_gap_artifacts = {
                    "gap_report.json",
                    "gap_report_pre_decompose.json",
                }
                if p.name in protected_gap_artifacts and out_dir_resolved and p.parent == out_dir_resolved:
                    return (
                        f"replace_in_file failed: {p.name} is a submitted gap-analysis artifact and "
                        "must not be edited directly. Build the updated report in memory, write it "
                        "to gap_report_draft.json, then call submit_gap_report(report_file=...). "
                        "submit_gap_report is the only tool allowed to create or overwrite the "
                        "official gap_report.json and recompute score/recommended_patch_order."
                    )
                if not p.exists():
                    return f"replace_in_file failed: file not found: {file_path}"
                original = p.read_text(encoding="utf-8")
                count = original.count(old_str)
                if count == 0:
                    return (
                        "replace_in_file failed: old_str not found in file.\n"
                        "Re-read the file with read_file and copy the exact text to replace."
                    )
                if count > 1:
                    return (
                        f"replace_in_file failed: old_str matches {count} locations — "
                        "provide more surrounding context to make it unique."
                    )
                backup = _skill_backup_path(p, root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_text(original, encoding="utf-8")
                updated = original.replace(old_str, new_str, 1)
                p.write_text(updated, encoding="utf-8")
                self_ref._append_diff(file_path, original, updated)
                return f"Replaced 1 occurrence in {file_path} (backup: {backup})"
            except Exception as exc:
                return f"replace_in_file failed: {exc}"

        return FunctionTool(
            name="replace_in_file",
            description=(
                "Replace an exact string in a file with a new string. "
                "PREFERRED over apply_patch for editing existing skill content. "
                "old_str must match exactly one location in the file (copy verbatim "
                "from read_file output). Automatically backs up the file before editing."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path (absolute or relative to project root)"},
                    "old_str": {"type": "string", "description": "Exact string to find (must be unique in the file)"},
                    "new_str": {"type": "string", "description": "String to replace it with"},
                },
                "required": ["path", "old_str", "new_str"],
            },
            on_invoke_tool=replace_in_file_handler,
            strict_json_schema=False,
        )

    def _make_apply_patch_tool(self, out_dir: str | None = None):
        editor = LocalFileEditor(self.project_root, diff_callback=self._append_diff)
        root = self.project_root
        self_ref = self
        out_dir_resolved = Path(out_dir).resolve() if out_dir else None

        async def apply_patch_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError as exc:
                return f"apply_patch failed: bad JSON — {exc}"
            operations = args.get("operations", [])

            _OP_MAP = {"create": "create_file", "update": "update_file", "delete": "delete_file"}
            protected_gap_artifacts = {"gap_report.json", "gap_report_pre_decompose.json"}

            results = []
            for op_dict in operations:
                # Accept both 'operation_type' (preferred) and 'type' (legacy) keys.
                raw_type = op_dict.get("operation_type") or op_dict.get("type") or "update"
                sdk_type = _OP_MAP.get(raw_type, raw_type)

                # ── Update path: prefer exact-string old_str/new_str (mirrors
                # replace_in_file).  LLMs are far more reliable at producing
                # exact-match strings than well-formed unified diffs.
                if sdk_type == "update_file" and op_dict.get("old_str") is not None:
                    old_str = op_dict["old_str"]
                    new_str = op_dict.get("new_str", "")
                    raw_path = op_dict.get("path", "")
                    if not raw_path:
                        results.append("[failed] missing 'path'")
                        continue
                    p = Path(raw_path)
                    if not p.is_absolute():
                        p = root / raw_path
                    p = p.resolve()
                    if not str(p).startswith(str(root.resolve())):
                        results.append(f"[failed] {raw_path}: path escapes project directory")
                        continue
                    if (
                        p.name in protected_gap_artifacts
                        and out_dir_resolved
                        and p.parent == out_dir_resolved
                    ):
                        results.append(
                            f"[failed] {raw_path}: {p.name} is a submitted gap-analysis "
                            "artifact and must not be edited directly. Use "
                            "submit_gap_report(report_file=...) instead."
                        )
                        continue
                    if not p.exists():
                        results.append(f"[failed] {raw_path}: file not found")
                        continue
                    try:
                        original = p.read_text(encoding="utf-8")
                    except Exception as exc:
                        results.append(f"[failed] {raw_path}: cannot read — {exc}")
                        continue
                    count = original.count(old_str)
                    if count == 0:
                        snippet = (old_str.splitlines() or [""])[0][:80]
                        results.append(
                            f"[failed] {raw_path}: old_str not found in file. "
                            f"Re-read the file with read_file and copy the exact "
                            f"text to replace. (First line of attempted old_str: {snippet!r})"
                        )
                        continue
                    if count > 1:
                        results.append(
                            f"[failed] {raw_path}: old_str matches {count} locations — "
                            f"provide more surrounding context to make it unique."
                        )
                        continue
                    updated = original.replace(old_str, new_str, 1)
                    try:
                        p.write_text(updated, encoding="utf-8")
                    except Exception as exc:
                        results.append(f"[failed] {raw_path}: cannot write — {exc}")
                        continue
                    # Wire the diff_callback for trajectory diff logging,
                    # matching replace_in_file's behavior.
                    try:
                        self_ref._append_diff(str(p), old_str, new_str)
                    except Exception:
                        pass
                    delta = len(updated) - len(original)
                    results.append(
                        f"[success] {raw_path}: replaced 1 occurrence ({delta:+d} chars)"
                    )
                    continue

                # ── Legacy SDK path: create / delete / update-with-diff ──
                operation = ApplyPatchOperation(
                    type=sdk_type,
                    path=op_dict.get("path", ""),
                    diff=op_dict.get("diff"),
                )
                if sdk_type == "create_file":
                    result = editor.create_file(operation)
                elif sdk_type == "update_file":
                    result = editor.update_file(operation)
                elif sdk_type == "delete_file":
                    result = editor.delete_file(operation)
                else:
                    results.append(f"[failed] unknown operation_type '{raw_type}' for {operation.path}")
                    continue
                results.append(f"[{result.status}] {operation.path}: {result.output}")

            return "\n".join(results) if results else "No operations performed"

        return FunctionTool(
            name="apply_patch",
            description=(
                "Apply a batch of file operations (create / update / delete) in one call.\n"
                "\n"
                "For operation_type='update', PREFER the exact-string replace mode: provide "
                "`old_str` (a verbatim unique snippet from the current file, copied from a "
                "recent read_file output) and `new_str` (the replacement).  This is the most "
                "reliable editing path and supersedes replace_in_file when batching is useful "
                "(e.g. one synthesis pass touching multiple sections).\n"
                "\n"
                "Legacy unified-diff mode (provide `diff` without old_str/new_str) is still "
                "supported but LLMs frequently produce malformed diffs.\n"
                "\n"
                "For operation_type='create', provide `diff` containing the full file body. "
                "For operation_type='delete', provide only the path."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation_type": {
                                    "type": "string",
                                    "enum": ["create", "update", "delete"],
                                },
                                "path":    {"type": "string"},
                                "old_str": {"type": "string"},
                                "new_str": {"type": "string"},
                                "diff":    {"type": "string"},
                            },
                            "required": ["operation_type", "path"],
                        },
                    },
                },
                "required": ["operations"],
            },
            on_invoke_tool=apply_patch_handler,
            strict_json_schema=False,
        )

    def _make_shell_tool(self):
        root = self.project_root

        async def shell_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            args = json.loads(args_json)
            commands = args["commands"]
            timeout_sec = (args.get("timeout_ms", 30000)) / 1000.0
            outputs = []
            for cmd in commands:
                try:
                    proc = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True,
                        timeout=timeout_sec, cwd=str(root), env={**os.environ},
                    )
                    out = f"$ {cmd}\n"
                    if proc.stdout:
                        out += proc.stdout
                    if proc.stderr:
                        out += f"[stderr] {proc.stderr}"
                    out += f"[exit {proc.returncode}]"
                    outputs.append(out)
                except subprocess.TimeoutExpired:
                    outputs.append(f"$ {cmd}\n[timeout after {timeout_sec}s]")
                    break
            return "\n\n".join(outputs) or "No output."

        return FunctionTool(
            name="shell",
            description=(
                "Run shell commands in the project directory.  Use to verify code "
                "patterns, check output files, or test a patch."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "commands":   {"type": "array", "items": {"type": "string"}},
                    "timeout_ms": {"type": "integer"},
                },
                "required": ["commands"],
            },
            on_invoke_tool=shell_handler,
            strict_json_schema=False,
        )

    def _make_generate_viz_tool(self):
        root = self.project_root

        async def generate_viz_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            try:
                import json as _json
                args = _json.loads(args_json)
                strong_path    = Path(args["strong_structure_json"])
                weak_path      = Path(args["weak_structure_json"])
                gap_path_arg   = args.get("gap_report_json")
                out_path_arg   = args.get("output_path")
                fmt            = args.get("fmt", "png")

                if not strong_path.is_absolute():
                    strong_path = root / strong_path
                if not weak_path.is_absolute():
                    weak_path = root / weak_path

                from execution_structure import ExecutionStructure, GapReport
                from execution_visualizer import visualize_comparison, visualize_structure

                strong = ExecutionStructure.model_validate(
                    _json.loads(strong_path.read_text(encoding="utf-8"))
                )
                weak = ExecutionStructure.model_validate(
                    _json.loads(weak_path.read_text(encoding="utf-8"))
                )

                gap_report = None
                if gap_path_arg:
                    gp = Path(gap_path_arg)
                    if not gp.is_absolute():
                        gp = root / gp
                    if gp.exists():
                        gap_report = GapReport.model_validate(
                            _json.loads(gp.read_text(encoding="utf-8"))
                        )

                out_dir = strong_path.parent
                if out_path_arg:
                    out_path = Path(out_path_arg)
                    if not out_path.is_absolute():
                        out_path = root / out_path
                else:
                    out_path = out_dir / f"comparison.{fmt}"

                rendered = visualize_comparison(
                    strong, weak, out_path,
                    gap_report=gap_report,
                    title=(
                        f"Comparison: {strong.agent_id} (strong) "
                        f"vs {weak.agent_id} (weak)"
                    ),
                )
                visualize_structure(strong, out_dir / f"strong_structure.{fmt}",
                                    title=f"{strong.agent_id} — execution structure")
                visualize_structure(weak, out_dir / f"weak_structure.{fmt}",
                                    title=f"{weak.agent_id} — execution structure")
                return (
                    f"Visualizations written:\n"
                    f"  {rendered}\n"
                    f"  {out_dir / f'strong_structure.{fmt}'}\n"
                    f"  {out_dir / f'weak_structure.{fmt}'}"
                )
            except Exception as exc:
                import traceback
                return f"generate_viz failed: {exc}\n{traceback.format_exc()}"

        return FunctionTool(
            name="generate_viz",
            description=(
                "Render annotated comparison visualization from ExecutionStructure JSONs "
                "and an optional gap_report.json.  Writes PNG/SVG to the structures directory."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "strong_structure_json": {"type": "string"},
                    "weak_structure_json":   {"type": "string"},
                    "gap_report_json":       {"type": "string"},
                    "output_path":           {"type": "string"},
                    "fmt":                   {"type": "string", "enum": ["png", "svg", "pdf"]},
                },
                "required": ["strong_structure_json", "weak_structure_json"],
            },
            on_invoke_tool=generate_viz_handler,
            strict_json_schema=False,
        )


__all__ = [
    "CaseContext",
    "StructuralDistance",
    "compute_structural_distance",
    "graph_edit_distance",
    "normalised_ged",
    "LocalFileEditor",
    "_skill_backup_path",
    "_file_role",
    "_expand_skill_listing",
    "FAIL_WEIGHT",
    "PASS_WEIGHT",
    "SCORE_THRESHOLD",
    "GAP_FILTER_MODE",
    "BINARY_MIN_FAILED",
    "BINARY_MIN_PASSED",
    "GapAgentBase",
]
