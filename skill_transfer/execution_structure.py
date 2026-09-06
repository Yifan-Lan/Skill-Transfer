"""
Data models for the structural trajectory abstraction pipeline.

Defines the schema for ExecutionStructure (a typed, ordered list of
ProcedureNodes derived from a raw agent trajectory) and GapReport (a
structured comparison of two ExecutionStructures).

Taxonomy design
───────────────
We use a two-level category scheme:

  category_tier1  5 universal categories + "other" that apply across ALL task
                  domains (document processing, scientific analysis, engineering
                  simulation, game AI, network analysis, etc.)

  category_tier2  A domain-specific label generated freely by the LLM, e.g.
                  "borderless_table_fallback", "bgp_route_extraction",
                  "stl_binary_parsing".  Not pre-defined — the abstractor
                  infers it from the trajectory context.

This two-level design lets the gap analyser use tier1 for rule-based,
domain-agnostic matching while tier2 carries the human-readable semantic
detail needed for skill patch hints.
"""

from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Tier-1 taxonomy  (universal, pre-defined)
# ──────────────────────────────────────────────────────────────────────────────
#
# Covers the procedural roles that appear in virtually every task type:
#
#   setup    — All preparatory work before the primary action: activating skills,
#              inspecting inputs, understanding file formats, surveying problem
#              structure, decomposing the task into sub-goals, initialising
#              environments.  Analogous to the "plan" phase in ReAct.
#              Examples: inspect PDF pages, parse STL binary header, understand
#              BGP log format, read game-state JSON.
#
#   execute  — The primary domain action(s): extraction, computation,
#              transformation, generation, analysis, filtering.  This is the
#              "doing" phase where the main work happens.
#              Examples: extract tables, compute mass from geometry, run
#              simulation step, parse dialogue script, solve optimisation.
#
#   validate — Checking correctness, completeness, or quality of intermediate
#              or final results BEFORE committing to them.  May occur multiple
#              times in a trajectory (e.g. after extraction AND after output).
#              Examples: coverage check (all labeled tables found?), verify
#              citation exists in a database, check graph JSON schema validity.
#
#   recover  — Any response to failure: fallback strategies, alternative
#              approaches, error handling, debugging, retrying with different
#              parameters, restructuring a failed attempt.
#              Examples: borderless-table word-coord reconstruction, retry API
#              with exponential backoff, fix bug in generated code, try a
#              different parsing library.
#
#   output   — Saving, writing, formatting, or reporting final artefacts.
#              Includes any post-save sanity read-back ("verify" in the old
#              taxonomy).
#              Examples: save CSV, write JSON result, print summary report,
#              render D3 visualisation, return answer string.
#
#   other    — Genuinely task-specific steps that do not fit any of the above
#              five.  Use sparingly; most steps should fit one of the five.
#              Examples: submit answer to judge API, send notification, update
#              shared state in a multi-agent pipeline.

Tier1Type = Literal["setup", "execute", "validate", "recover", "output", "other"]

OutcomeType = Literal["success", "error", "partial", "skipped"]
GapType     = Literal["missing_node", "weak_node", "order_difference"]
SeverityType = Literal["high", "medium", "low"]


# ──────────────────────────────────────────────────────────────────────────────
# ExecutionStructure models
# ──────────────────────────────────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """A single tool invocation within a procedural node."""

    tool: str = Field(
        description="Tool name: shell, read_file, write_file, activate_skill, etc."
    )
    purpose: str = Field(
        description="One-sentence description of why this call was made."
    )
    code_snippet: Optional[str] = Field(
        default=None,
        description="Key code or command (truncated to ~200 chars if long).",
    )
    observation: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim snippet of the tool's actual output, truncated to ≤300 chars. "
            "Null for activate_skill calls."
        ),
    )
    outcome: OutcomeType = Field(description="Whether the tool call succeeded.")
    output_summary: str = Field(description="Key result in one or two sentences.")
    aborted_run: bool = Field(
        default=False,
        description=(
            "True if this tool call triggered a pipeline abort (e.g. output too large "
            "for the token budget). The [PIPELINE ABORT] marker will appear in the "
            "trajectory observation for this call."
        ),
    )


class ProcedureNode(BaseModel):
    """A discrete procedural step in the agent's execution.

    Fields
    ──────
    category_tier1   Universal procedural role (setup / execute / validate /
                     recover / output / other).  Used for rule-based gap
                     detection across task domains.

    category_tier2   Domain-specific semantic label generated by the abstractor
                     LLM, e.g. "page_coverage_check", "stl_mass_computation",
                     "api_retry_with_backoff".  Not pre-defined.

    completes_when   A concrete criterion that defines successful completion of
                     this node.  Enables the gap analyser to detect *weak*
                     nodes — cases where the weak agent has the tier1 category
                     but its completion criterion is less rigorous than the
                     strong agent's.
                     Example: "when every page whose text contains 'Table N'
                     has a corresponding row in extracted_tables".

    produces         Short labels for data artefacts or state this step creates.
                     Used for lightweight data-flow tracking.
                     Example: ["candidate_tables", "page_count"]

    consumes         Short labels for data artefacts or context this step
                     requires as input.
                     Example: ["pdf_path", "skill_instructions"]
    """

    id: str = Field(description="Unique identifier, e.g. 'node_001'.")
    step_index: int = Field(description="Zero-based position in the ordered list.")
    label: str = Field(
        description="Short semantic name in snake_case, e.g. 'page_coverage_check'."
    )

    # Two-level category
    category_tier1: Tier1Type = Field(
        description="Universal procedural role: setup, execute, validate, recover, output, or other."
    )
    category_tier2: str = Field(
        description=(
            "Domain-specific semantic label (free text, snake_case). "
            "Generated by the abstractor LLM. "
            "Examples: 'borderless_table_fallback', 'bgp_route_extraction', "
            "'stl_binary_header_parse', 'citation_database_lookup'."
        )
    )

    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Tool invocations that make up this step.",
    )
    agent_intent: str = Field(
        description="Why the agent performed this step (from its reasoning/thought)."
    )

    # Completion semantics
    completes_when: str = Field(
        description=(
            "Concrete criterion for successful completion of this node. "
            "Used to detect weak nodes where the category is present but "
            "the completion bar is lower than the strong agent's. "
            "Example: 'when every page labelled Table N has a saved CSV'."
        )
    )

    # Lightweight data-flow
    produces: list[str] = Field(
        default_factory=list,
        description=(
            "Short snake_case labels for data artefacts or state produced by "
            "this step. Example: ['candidate_tables', 'page_count']."
        ),
    )
    consumes: list[str] = Field(
        default_factory=list,
        description=(
            "Short snake_case labels for data artefacts or context consumed by "
            "this step. Example: ['pdf_path', 'skill_instructions']."
        ),
    )

    outcome: OutcomeType = Field(description="Overall outcome of this step.")
    notes: Optional[str] = Field(
        default=None,
        description="Notable details: edge cases handled, important parameters, etc.",
    )
    branch_of: Optional[str] = Field(
        default=None,
        description="Parent node id if this step is a conditional sub-step or retry.",
    )
    skill_selection_reason: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim `reason` argument from the activate_skill tool call in this node. "
            "Null for nodes that do not call activate_skill."
        ),
    )
    run_aborted: bool = Field(
        default=False,
        description=(
            "True if the agent run was terminated by the pipeline during this node "
            "(i.e. a tool call in this node has aborted_run=true)."
        ),
    )
    abort_reason: Optional[str] = Field(
        default=None,
        description=(
            "The [PIPELINE ABORT] message from the trajectory observation of the "
            "tool call that triggered the abort. Null when run_aborted is false."
        ),
    )


class ExecutionStructure(BaseModel):
    """High-level execution structure abstracted from a raw agent trajectory."""

    agent_id: str = Field(description="Identifier for the agent (e.g. model name).")
    task: str = Field(description="One-line task description.")
    total_steps: int = Field(description="Number of nodes in this structure.")
    nodes: list[ProcedureNode] = Field(description="Ordered list of procedural steps.")
    outcome: OutcomeType = Field(description="Overall task outcome.")
    produced_files: list[str] = Field(
        default_factory=list,
        description="Output files the agent created.",
    )
    abstraction_notes: Optional[str] = Field(
        default=None,
        description="Notes on ambiguities or interpretation choices made during abstraction.",
    )

    # ── helpers ──────────────────────────────────────────────────────────────

    def tier1_categories(self) -> set[Tier1Type]:
        """Return the set of tier1 categories present in this structure."""
        return {node.category_tier1 for node in self.nodes}

    def tier2_labels(self) -> set[str]:
        """Return the set of tier2 labels present in this structure."""
        return {node.category_tier2 for node in self.nodes}

    def nodes_by_tier1(self, category: Tier1Type) -> list[ProcedureNode]:
        """Return all nodes with the given tier1 category."""
        return [n for n in self.nodes if n.category_tier1 == category]

    def nodes_by_tier2(self, label: str) -> list[ProcedureNode]:
        """Return all nodes with the given tier2 label (exact match)."""
        return [n for n in self.nodes if n.category_tier2 == label]

    def all_produced(self) -> set[str]:
        """Union of all produces labels across all nodes."""
        return {p for node in self.nodes for p in node.produces}

    def all_consumed(self) -> set[str]:
        """Union of all consumes labels across all nodes."""
        return {c for node in self.nodes for c in node.consumes}


# ──────────────────────────────────────────────────────────────────────────────
# GapReport models
# ──────────────────────────────────────────────────────────────────────────────

class GapEntry(BaseModel):
    """A single identified gap between strong and weak agent execution structures."""

    gap_id: str = Field(description="Unique identifier, e.g. 'gap_001'.")
    gap_type: GapType = Field(
        description=(
            "missing_node: entire tier1 category absent in weak agent. "
            "weak_node: tier1 category present but completes_when is less rigorous, "
            "or produces fewer artefacts. "
            "order_difference: step present but executed in wrong order."
        )
    )

    # Category references
    category_tier1: Tier1Type = Field(
        description="Tier1 category this gap concerns."
    )
    category_tier2: Optional[str] = Field(
        default=None,
        description="Tier2 label from the strong agent's node (if applicable).",
    )

    label: str = Field(description="Short name for this gap, in snake_case.")
    description: str = Field(
        description="What the strong agent did that the weak agent did not, concretely."
    )

    # Node references
    # Single-case mode: plain node id string (e.g. "node_005") or null.
    # Multi-case mode: dict mapping case_id → node_id (or null) for every
    # case in failed_case_ids + pass_case_ids.
    strong_node_ref: Optional[Union[str, dict[str, Optional[str]]]] = Field(
        default=None,
        description=(
            "Single-case: id of the strong agent's ProcedureNode. "
            "Multi-case: {case_id: node_id} for ALL cases in failed_case_ids + pass_case_ids."
        ),
    )
    weak_node_ref: Optional[Union[str, dict[str, Optional[str]]]] = Field(
        default=None,
        description=(
            "Single-case: id of the weak agent's ProcedureNode (for weak_node gaps). "
            "Multi-case: {case_id: node_id or null} for ALL cases in failed_case_ids + pass_case_ids."
        ),
    )
    weak_evidence: Optional[Union[str, dict[str, Optional[str]]]] = Field(
        default=None,
        description=(
            "Single-case: verbatim snippet from the weak trajectory. "
            "Multi-case: {case_id: verbatim snippet or null} for ALL cases in failed_case_ids + pass_case_ids."
        ),
    )
    strong_evidence: Optional[Union[str, dict[str, Optional[str]]]] = Field(
        default=None,
        description=(
            "Single-case: verbatim snippet from the strong trajectory showing what the strong agent did. "
            "Multi-case: {case_id: verbatim snippet or null} for ALL cases in failed_case_ids + pass_case_ids."
        ),
    )

    # Completion criterion gap (for weak_node gaps)
    strong_completes_when: Optional[str] = Field(
        default=None,
        description="The strong agent's completes_when for this step.",
    )
    weak_completes_when: Optional[str] = Field(
        default=None,
        description="The weak agent's completes_when (if weaker/absent).",
    )

    # Data-flow gap (optional)
    missing_produces: list[str] = Field(
        default_factory=list,
        description=(
            "Artefacts the strong agent's node produces that the weak agent's "
            "equivalent node does not."
        ),
    )

    skill_patch_hint: str = Field(
        description="What should be added or changed in the skill to address this gap."
    )
    severity: SeverityType = Field(
        description="high: directly caused task failure. medium: degraded quality. low: minor."
    )

    # ── Multi-case scoring fields (populated in MULTI mode only) ──────────────
    failed_case_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Cases where the weak agent FAILED and this gap was observed. "
            "Each contributes fail_weight to score. (Multi mode only.)"
        ),
    )
    pass_case_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Cases where the weak agent PASSED but this gap was still observed, "
            "indicating a latent robustness risk. "
            "Each contributes pass_weight to score. (Multi mode only.)"
        ),
    )
    score: Optional[float] = Field(
        default=None,
        description=(
            "Weighted gap score: (fail_weight × n_fail + pass_weight × n_pass) / total_cases. "
            "Gaps with score ≥ score_threshold are patched. (Multi mode only.)"
        ),
    )


class GapReport(BaseModel):
    """Structured comparison of two ExecutionStructures, identifying gaps."""

    strong_agent: str = Field(description="Agent id of the strong agent.")
    weak_agent: str = Field(description="Agent id of the weak agent.")
    outcome_summary: dict[str, str] = Field(
        description="{'strong': '...', 'weak': '...'} describing each agent's outcome."
    )
    gaps: list[GapEntry] = Field(description="All identified gaps, ordered by severity.")
    summary: str = Field(
        description="One-paragraph summary of the main differences and their cause."
    )
    recommended_patch_order: list[str] = Field(
        description="gap_ids in recommended patch priority order (most impactful first)."
    )

    # ── Multi-case scoring metadata (populated in MULTI mode only) ────────────
    fail_weight: Optional[float] = Field(
        default=None,
        description="Weight applied per failed case when computing gap scores. (Multi mode only.)",
    )
    pass_weight: Optional[float] = Field(
        default=None,
        description="Weight applied per passing case when computing gap scores. (Multi mode only.)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score for a gap to be included in recommended_patch_order. (Multi mode only.)",
    )
    total_cases: Optional[int] = Field(
        default=None,
        description="Total number of cases analysed in this batch. (Multi mode only.)",
    )
