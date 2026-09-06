#!/usr/bin/env python3
"""
TrajectoryAbstractor: converts a raw agent trajectory (markdown or json) into an
ExecutionStructure (typed, ordered procedural nodes).

Implemented as an agentic loop using the OpenAI Agents SDK.  The agent:
  1. Reads the trajectory (provided in the prompt, or via read_file).
  2. Optionally uses shell or read_file for deeper inspection.
  3. Optionally calls validate_draft() to self-correct schema errors.
  4. Calls submit_structure() with the final JSON when satisfied.

Public API is unchanged — run_abstractor.py needs no modifications:

    abstractor = TrajectoryAbstractor(model="gpt-5.4")
    structure = abstractor.abstract(
        trajectory_md=open("trajectory.md").read(),
        agent_id="gpt-5.4",
        task="Extract all tables from PDF and save as CSV",
        output_path=Path("structures/strong_structure.json"),
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool
from run_skill_agent import (
    _get_event_type,
    _extract_text_delta,
    _extract_tool_call_name_and_args,
    _looks_like_tool_event,
)

from execution_structure import ExecutionStructure

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repository root


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building blocks
# ──────────────────────────────────────────────────────────────────────────────

TIER1_GUIDE = textwrap.dedent("""
    ## Tier-1 Categories  (use EXACTLY one of these 6 values)

    These are universal procedural roles that apply across ALL task domains —
    document processing, scientific analysis, engineering simulation, network
    analysis, game AI, web tasks, security analysis, etc.

    setup
        All preparatory work BEFORE the primary action.
        Includes: activating skills, inspecting input files/formats, understanding
        data structure, surveying the problem space, decomposing the task into
        sub-goals, initialising environments, reading configuration.
        Examples across domains:
          - PDF task:     inspect page count and layout with pdfplumber
          - STL task:     parse binary STL header to understand mesh format
          - BGP task:     read routing log and identify anomaly patterns
          - Cipher task:  understand cipher algorithm before attempting decryption
          - Game task:    read game-state JSON to understand district adjacency rules

    execute
        The PRIMARY domain action — the core "doing" work of the task.
        Includes: extraction, computation, transformation, analysis, generation,
        filtering/rejection of false positives, classification, solving.
        This is the largest category; a trajectory may have multiple execute nodes
        for distinct sub-tasks (e.g. extract table A, then extract table B).
        Examples:
          - PDF task:     call extract_tables() on each page
          - STL task:     compute mass from triangle mesh geometry
          - Citation:     query each citation against a reference database
          - Game task:    evaluate adjacency bonus for each district placement
          - Simulation:   run control loop for each timestep

    validate
        Checking correctness, completeness, or quality of results — whether
        intermediate or final — BEFORE committing to them.
        Key signal: the agent asks "have I found everything?" or "is this result
        correct?" before moving on.  Distinct from verify (a post-save read-back):
        validate is a deliberate coverage/quality gate.
        Examples:
          - PDF task:     check that every page with "Table N" text has a saved CSV
          - Citation:     confirm all suspicious citations were checked, not just some
          - Network:      verify that all BGP peers are accounted for in the report
          - Code gen:     run unit tests on generated code before returning it
          - 3D task:      cross-check computed mass against expected range

    recover
        Any response to failure, uncertainty, or an insufficient primary result.
        Includes: fallback strategies, alternative approaches, debugging, retrying
        with different parameters, restructuring a failed attempt, error handling.
        Key signal: the agent detects a problem and tries something different.
        Examples:
          - PDF task:     use extract_words() + coordinate grouping when
                          extract_tables() returns nothing for a page that visually
                          has a table
          - API task:     retry with exponential backoff after rate-limit error
          - Code task:    fix a bug in generated code after a test failure
          - Network:      fall back to heuristic detection when ML model fails
          - Parsing:      try a secondary regex pattern after the primary fails

    output
        Saving, writing, formatting, or reporting FINAL artefacts.
        Includes: writing files to disk, printing/returning the final answer,
        rendering a visualisation, formatting a report, any post-save read-back
        to confirm the output is correct (the old "verify" role).
        Examples:
          - PDF task:     save extracted tables as CSV; read back to confirm content
          - Simulation:   write simulation results to JSON; print summary
          - Web task:     render D3 chart and save as HTML; open in browser to verify
          - Analysis:     return structured JSON answer to the judge
          - Game task:    output optimal district placement as a list

    other
        Genuinely task-specific steps that do not fit any of the above five.
        Use sparingly — most steps should map to one of the five above.
        If you find yourself reaching for "other" frequently, reconsider whether
        the step is a sub-part of an existing node rather than its own node.
        Examples of legitimate use:
          - Submit answer to an external judge API (not just saving locally)
          - Send a notification or webhook in a multi-agent pipeline
          - Update shared state that another agent will read
""").strip()


TIER2_GUIDE = textwrap.dedent("""
    ## Tier-2 Label  (free text, snake_case, generated by you)

    The tier2 label is a SHORT, domain-specific semantic name for this specific
    step — not a general category.  You invent it based on what the step actually
    does.  It should be specific enough that a developer reading it immediately
    knows what happened.

    Rules:
    - snake_case, ≤5 words, descriptive
    - Reflects the specific action, not the general category
    - Include the domain context if helpful

    Good tier2 labels:
        "page_coverage_check"         (validate step in PDF task)
        "borderless_table_fallback"   (recover step in PDF task)
        "stl_binary_header_parse"     (setup step in 3D scan task)
        "bgp_peer_route_extraction"   (execute step in network task)
        "citation_database_lookup"    (execute step in citation task)
        "district_adjacency_scoring"  (execute step in game task)
        "control_loop_timestep"       (execute step in simulation task)
        "api_rate_limit_retry"        (recover step in API task)
        "json_output_schema_check"    (validate step in any task)

    Bad tier2 labels (too generic):
        "main_task", "step_1", "processing", "action"
""").strip()


COMPLETES_WHEN_GUIDE = textwrap.dedent("""
    ## completes_when  (completion criterion)

    A single concrete sentence describing what must be true for this node to be
    considered successfully completed.  This is the most important field for
    gap analysis: it lets us detect *weak nodes* — cases where the weak agent
    has the tier1 category but its completion bar is lower than the strong agent's.

    Rules:
    - Specific and falsifiable ("when X has been done" not "when processing is done")
    - Grounded in the trajectory (what did the agent actually check or produce?)
    - For execute: what data/artefacts must exist?
    - For validate: what coverage condition must hold?
    - For recover: what is the success condition for the fallback?

    Examples across domains:
        setup:    PDF task:        "when all pages have been scanned and the
                                    table-per-page count is known"
                  Spreadsheet task: "when the workbook sheets and relevant cell
                                    ranges have been identified"
                  API task:         "when the available endpoints and required
                                    auth parameters are known"

        execute:  PDF task:        "when all detected tables have been extracted
                                    and false positives excluded"
                  Spreadsheet task: "when all required formulas have been
                                    entered in the specified cell ranges"
                  Code-gen task:    "when all requested functions have been
                                    implemented and pass a syntax check"

        validate: PDF task:        "when every page with 'Table N' text has a
                                    saved CSV in the output directory"
                  Spreadsheet task: "when every target cell range contains a
                                    formula and produces a numeric result"
                  Data task:        "when all expected rows/columns are
                                    accounted for with no missing values"

        recover:  PDF task:        "when the missed borderless table has been
                                    reconstructed from word coordinates"
                  API task:         "when the retry succeeded after rate-limit
                                    and the response is valid"
                  Code task:        "when the failing test now passes after
                                    the bug fix"

        output:   PDF task:        "when CSVs have been saved and read back
                                    to confirm expected row/column structure"
                  Spreadsheet task: "when the modified workbook has been saved
                                    and the formula results verified"
                  Analysis task:    "when the result JSON has been written and
                                    the key metrics match expected ranges"
""").strip()


DATAFLOW_GUIDE = textwrap.dedent("""
    ## produces / consumes  (lightweight data flow)

    List short snake_case labels for the data artefacts or state variables this
    step creates (produces) or requires as input (consumes).

    Purpose: enables the gap analyser to detect when a weak agent's execution
    node produces fewer artefacts than the strong agent's — e.g. the weak agent's
    execute node produces only ["table_001_csv"] while the strong agent's produces
    ["table_001_csv", "table_002_csv"].

    Rules:
    - Short, snake_case, descriptive
    - Abstract enough to be meaningful (not a full file path)
    - Include ALL significant artefacts, not just the "main" output

    Examples across domains:
        PDF task:
          setup node:    produces: ["page_count", "per_page_table_counts"]
                         consumes: ["input_pdf", "skill_instructions"]
          execute node:  produces: ["candidate_tables", "false_positive_flags"]
                         consumes: ["page_count", "per_page_table_counts"]
          validate node: produces: ["coverage_report", "missing_table_pages"]
                         consumes: ["candidate_tables", "per_page_table_counts"]
          recover node:  produces: ["reconstructed_table"]
                         consumes: ["missing_table_pages", "input_pdf"]
          output node:   produces: ["table_001_csv", "table_002_csv"]
                         consumes: ["candidate_tables", "reconstructed_table"]

        Spreadsheet task:
          setup node:    produces: ["sheet_names", "target_cell_ranges"]
                         consumes: ["input_workbook", "skill_instructions"]
          execute node:  produces: ["formulas_entered", "computed_values"]
                         consumes: ["sheet_names", "target_cell_ranges"]
          validate node: produces: ["formula_error_check", "missing_cells"]
                         consumes: ["formulas_entered", "target_cell_ranges"]
          output node:   produces: ["saved_workbook"]
                         consumes: ["formulas_entered"]

        API / data task:
          setup node:    produces: ["endpoint_list", "auth_config"]
                         consumes: ["api_docs", "skill_instructions"]
          execute node:  produces: ["raw_responses", "parsed_records"]
                         consumes: ["endpoint_list", "auth_config"]
          recover node:  produces: ["retried_responses"]
                         consumes: ["rate_limit_errors", "endpoint_list"]
          output node:   produces: ["result_json"]
                         consumes: ["parsed_records"]
""").strip()


SCHEMA_SPEC = textwrap.dedent("""
    ## Output Schema

    The final JSON must match this structure exactly:

    {
      "agent_id": "<string>",
      "task": "<one-line task description>",
      "total_steps": <integer — must equal len(nodes)>,
      "nodes": [ <ProcedureNode>, ... ],
      "outcome": "<success|error|partial|skipped>",
      "produced_files": ["<filename>", ...],   // ONLY user-requested deliverables (e.g. .xlsx, .csv, .json report).
                                                 // DO NOT include execution scripts (.py, .sh), intermediate dumps,
                                                 // or any file the agent created just to run code. If the task asked
                                                 // for a single spreadsheet, list only that spreadsheet.
      "abstraction_notes": "<optional notes on ambiguities>"
    }

    Each ProcedureNode:
    {
      "id": "node_NNN",            // zero-padded 3-digit, starting node_000
      "step_index": <int>,          // 0-based
      "label": "<snake_case>",      // short semantic name, same as category_tier2
      "category_tier1": "<one of: setup, execute, validate, recover, output, other>",
      "category_tier2": "<domain-specific snake_case label>",
      "tool_calls": [
        {
          "tool": "<tool name>",
          "purpose": "<why>",
          "code_snippet": "<#N reference into the Tool Call Reference table; never write the raw code>",
          "observation": null,
          "outcome": "<success|error|partial|skipped>",
          "output_summary": "<1-2 sentence result>"
        }
      ],
      "agent_intent": "<why the agent did this step>",
      "completes_when": "<concrete completion criterion>",
      "produces": ["<artefact_label>", ...],
      "consumes": ["<artefact_label>", ...],
      "outcome": "<success|error|partial|skipped>",
      "notes": "<notable details or null>",
      "branch_of": "<parent node id or null>",
      "skill_selection_reason": "<verbatim `reason` arg from activate_skill, or null>"
    }

    IMPORTANT: if a node's tool_calls contains an activate_skill call, copy its
    `reason` argument VERBATIM into skill_selection_reason.  For all other nodes
    set skill_selection_reason to null.

    ## observation field
    Always set to null. The pipeline fills this field automatically from the
    raw trajectory after you submit, so emitting it verbatim only wastes
    output tokens. (`output_summary` below is where you should put the
    distilled result.)

    ## output_summary field
    For shell/python computation steps, the summary MUST include the actual numeric
    values produced (e.g. "Extracted y=[46011,54119,56436,...]; slope=273.28,
    intercept=54244"). Do not paraphrase numbers — copy them from the output directly.
""").strip()


GRANULARITY_RULES = textwrap.dedent("""
    ## Granularity Rules

    1. Each tool call (or logical group of tightly-coupled calls) with a DISTINCT
       purpose gets its own node.  Do NOT merge a page-coverage probe and a table
       extraction into one node even if they appear in the same shell block.

    2. A retry or alternative strategy is its own node (category_tier1 = recover)
       with branch_of pointing to the node it retries.

    3. A node may have multiple tool_calls only when they are sub-steps of ONE
       logical action (e.g. extract words AND group by coordinates — both part of
       one fallback reconstruction).

    4. Aim for 5–15 nodes for a typical trajectory.  Do not create trivial nodes
       for boilerplate (import statements, variable assignments).

    5. skill activation ALWAYS gets its own node (category_tier1 = setup,
       category_tier2 = "skill_activation").
""").strip()


EXAMPLE_NODE = textwrap.dedent("""
    ## Example Node (for formatting reference only — do not copy content)

    {
      "id": "node_004",
      "step_index": 4,
      "label": "page_coverage_check",
      "category_tier1": "validate",
      "category_tier2": "page_coverage_check",
      "tool_calls": [
        {
          "tool": "shell",
          "purpose": "Check whether every page that mentions a Table caption has a corresponding extracted table",
          "code_snippet": "#12",
          "observation": null,
          "outcome": "success",
          "output_summary": "Page 4 has 'Table 5-1' in text but extract_tables() returned empty — needs fallback"
        }
      ],
      "agent_intent": "Ensure no labeled table pages are missed before committing to final output",
      "completes_when": "when every page whose raw text contains 'Table N' has a non-empty entry in the extracted tables list",
      "produces": ["coverage_report", "missing_table_pages"],
      "consumes": ["per_page_table_candidates", "per_page_text"],
      "outcome": "success",
      "notes": "Page 4 identified as needing borderless reconstruction",
      "branch_of": null
    }
""").strip()


# System prompt: all static guidance
_ABSTRACTOR_SYSTEM_PROMPT = f"""You are an expert at abstracting LLM agent execution trajectories into structured procedural representations.

Your task is to read the trajectory in the user message and produce a structured ExecutionStructure JSON by calling submit_structure().

Work methodically:
1. Read the full trajectory provided in the user message.
2. If you need to re-examine specific parts of the trajectory file or inspect related output files, use read_file() or shell().
3. Draft the ExecutionStructure nodes, applying the schema and granularity rules below.
4. Optionally call validate_draft() on your draft to catch schema errors — fix any reported errors before submitting.
5. Call submit_structure() with the final JSON.

You MUST call submit_structure() — do not output the JSON as plain text.

code_snippet field: the user message contains a numbered Tool Call Reference table.
For each ToolCallRecord, set code_snippet to "#N" where N is the matching index
from that table (e.g. "#3").  Do NOT copy raw code — the system fills in the
actual code automatically.

observation field: always set to null.  The system fills it from the raw
trajectory after you submit (matched by tool-call order).  Do NOT copy raw
output; emitting it only wastes output tokens.  Write your distilled result
into `output_summary` instead.

WRITE-FILE + EXECUTE PATTERN: when an agent writes a script via write_file then
runs it with shell, both tool calls belong to the same node.  Use the write_file
index for code_snippet of that tool call record (e.g. "#7"), and the shell index
for the execute record (e.g. "#8").  The system will fill in the correct code for
each.

{TIER1_GUIDE}

{TIER2_GUIDE}

{COMPLETES_WHEN_GUIDE}

{DATAFLOW_GUIDE}

{GRANULARITY_RULES}

{SCHEMA_SPEC}

{EXAMPLE_NODE}
"""


def _build_user_message(
    trajectory_md: str,
    agent_id: str,
    task: str,
    reference_structure_json: str | None = None,
    reference_trajectory_md: str | None = None,
) -> str:
    parts: list[str] = [f"Agent ID: {agent_id}\nTask: {task}\n"]

    if reference_structure_json or reference_trajectory_md:
        parts.append(
            "## Strong Agent Reference\n"
            "The following material comes from the strong (successful) agent on the same task.\n"
            "Use it to:\n"
            "  1. Align tier2 labels — use the SAME category_tier2 label for steps that are\n"
            "     semantically equivalent to steps in the reference structure.\n"
            "  2. Understand the complete intended procedure — the weak agent's trajectory\n"
            "     may be missing or weaker on some of these steps.\n"
        )

    if reference_structure_json:
        parts.append(
            "### Reference ExecutionStructure (strong agent)\n"
            f"```json\n{reference_structure_json}\n```\n"
        )

    from trajectory_parser import parse_tool_calls, compact_trajectory, build_reference_table

    if reference_trajectory_md:
        ref_calls   = parse_tool_calls(reference_trajectory_md)
        ref_compact = compact_trajectory(reference_trajectory_md, ref_calls)
        ref_ref     = build_reference_table(ref_calls)
        parts.append(
            "### Reference Trajectory (strong agent)\n"
            f"## Reference Tool Call Table\n{ref_ref}\n\n"
            f"--- STRONG TRAJECTORY START ---\n"
            f"{ref_compact}\n"
            f"--- STRONG TRAJECTORY END ---\n"
        )

    traj_calls   = parse_tool_calls(trajectory_md)
    traj_compact = compact_trajectory(trajectory_md, traj_calls)
    traj_ref     = build_reference_table(traj_calls)

    parts.append(
        f"## Tool Call Reference (use #N for code_snippet)\n{traj_ref}\n\n"
        f"## Trajectory to Abstract\n"
        f"--- TRAJECTORY START ---\n"
        f"{traj_compact}\n"
        f"--- TRAJECTORY END ---\n\n"
        f"For code_snippet, write only \"#N\" (e.g. \"#2\") from the reference table.\n"
        f"Now produce the ExecutionStructure JSON by calling submit_structure().\n"
        f"Ensure total_steps equals the number of nodes in the nodes array."
    )

    if reference_structure_json or reference_trajectory_md:
        parts.append(
            "\nIMPORTANT: Align tier2 labels with the reference structure for "
            "equivalent steps so that the downstream gap analyser can compare them directly."
        )

    return "\n".join(parts)


# Joint system prompt — used when both trajectories are abstracted together
_ABSTRACTOR_JOINT_SYSTEM_PROMPT = f"""You are an expert at abstracting LLM agent execution trajectories into structured procedural representations.

You will receive TWO trajectories — a STRONG agent (successful) and a WEAK agent (suboptimal/failed) — for the same task.
Produce an ExecutionStructure JSON for each by calling:
  - submit_strong_structure() for the strong agent
  - submit_weak_structure() for the weak agent

Work methodically:
1. The user message contains two Tool Call Reference tables (one per agent) with
   pre-extracted tool calls numbered #0, #1, … Use these to identify steps.
2. Identify the strong agent's procedural steps first, then the weak agent's.
3. Use CONSISTENT tier1/tier2 labels across both structures — if both agents
   performed the same kind of step, use the same category_tier2 label for it.
   This makes gap analysis easier downstream.
4. For code_snippet in each ToolCallRecord, write ONLY "#N" (e.g. "#3") — the index
   from the reference table.  Do NOT copy raw code.  The system fills it in.
5. Optionally call validate_draft() on either draft to catch schema errors.
6. Call submit_strong_structure() and submit_weak_structure() with the final JSONs.
   You MUST call both — do not output JSON as plain text.

Joint abstraction advantage: you can directly observe where the weak agent
diverges from the strong agent's procedure while labeling, producing more
semantically aligned structures for gap analysis.

{TIER1_GUIDE}

{TIER2_GUIDE}

{COMPLETES_WHEN_GUIDE}

{DATAFLOW_GUIDE}

{GRANULARITY_RULES}

{SCHEMA_SPEC}

{EXAMPLE_NODE}
"""


def _build_joint_user_message(
    strong_md: str,
    weak_md: str,
    strong_id: str,
    weak_id: str,
    task: str,
    strong_tool_calls: "list | None" = None,
    weak_tool_calls:   "list | None" = None,
) -> str:
    from trajectory_parser import parse_tool_calls, compact_trajectory, build_reference_table

    s_calls = strong_tool_calls if strong_tool_calls is not None else parse_tool_calls(strong_md)
    w_calls = weak_tool_calls   if weak_tool_calls   is not None else parse_tool_calls(weak_md)

    s_compact = compact_trajectory(strong_md, s_calls)
    w_compact = compact_trajectory(weak_md,   w_calls)

    s_ref = build_reference_table(s_calls)
    w_ref = build_reference_table(w_calls)

    return (
        f"Task: {task}\n\n"
        f"### Strong Agent ({strong_id}) — Tool Call Reference\n{s_ref}\n\n"
        f"--- STRONG AGENT TRAJECTORY ({strong_id}) START ---\n"
        f"{s_compact}\n"
        f"--- STRONG AGENT TRAJECTORY END ---\n\n"
        f"### Weak Agent ({weak_id}) — Tool Call Reference\n{w_ref}\n\n"
        f"--- WEAK AGENT TRAJECTORY ({weak_id}) START ---\n"
        f"{w_compact}\n"
        f"--- WEAK AGENT TRAJECTORY END ---\n\n"
        f"Now produce both ExecutionStructure JSONs.\n"
        f"For code_snippet, use only \"#N\" referencing the table above.\n"
        f"Call submit_strong_structure() for {strong_id} and "
        f"submit_weak_structure() for {weak_id}.\n"
        f"Use consistent tier2 labels where the agents performed the same step."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal result stores
# ──────────────────────────────────────────────────────────────────────────────

class _StructureStore:
    def __init__(self):
        self.structure: ExecutionStructure | None = None


class _PairStore:
    def __init__(self):
        self.strong: ExecutionStructure | None = None
        self.weak: ExecutionStructure | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────

_OBS_LIMIT      = 400   # final post-processing cap (smart-truncate above this)
_OBS_HEAD_KEEP  = 120   # head bytes kept by smart-truncate (typically `stderr:` header + offending-line preview)
_OBS_TAIL_KEEP  = 220   # tail bytes kept by smart-truncate (typically `<ErrorType>: ...` + exit_code line)
_OBS_ELLIPSIS   = "\n... [middle truncated] ...\n"

# Shell tool observations have the format:
#   Command: <cmd>\n(stdout|stderr|exit_code): <output>
# The "Command: <cmd>" prefix is redundant with `code_snippet` (which already
# captures the verbatim command) — it wastes the truncation budget and pushes
# the actual output (stdout / stderr / exit_code) past the cap. We strip the
# prefix in post-processing so the output portion survives, including final
# stderr lines that carry SyntaxError / Traceback / #VALUE! etc.
_SHELL_OUTPUT_MARKER_RE = re.compile(r"\n(stdout|stderr|exit_code):", re.IGNORECASE)


def _strip_shell_command_echo(obs: str) -> str:
    """For shell tool observations, drop the leading 'Command: <cmd>' echo.

    Returns the remainder starting at the first stdout: / stderr: / exit_code:
    marker. If the observation lacks any of those markers (no command-runner
    output) it is returned unchanged.
    """
    if not obs or not obs.startswith("Command:"):
        return obs
    m = _SHELL_OUTPUT_MARKER_RE.search(obs)
    if not m:
        return obs
    return obs[m.start():].lstrip("\n")


def _smart_truncate(
    obs: str,
    head: int = _OBS_HEAD_KEEP,
    tail: int = _OBS_TAIL_KEEP,
    limit: int = _OBS_LIMIT,
) -> str:
    """Head + tail truncation for shell-style observations.

    Python tracebacks place the signal at HEAD (`stderr:` / `File "<...>"`) and
    at TAIL (`<ErrorType>: message` + `exit_code: N`); the middle is usually
    Python re-printing the offending source line, which is already in
    `code_snippet`. Keeping head + tail preserves both anchor points while
    discarding the redundant middle.
    """
    if len(obs) <= limit:
        return obs
    return obs[:head].rstrip() + _OBS_ELLIPSIS + obs[-tail:].lstrip()


def _truncate_observations(structure: ExecutionStructure) -> None:
    """Normalise & cap every ToolCallRecord.observation in-place.

    - shell calls: strip the redundant "Command: <cmd>" prefix, then
      smart-truncate (head + tail) when the result exceeds _OBS_LIMIT.
    - other tools: simple head truncation to _OBS_LIMIT chars.
    """
    for node in structure.nodes:
        for tc in node.tool_calls:
            obs = tc.observation
            if not obs:
                continue
            if tc.tool == "shell":
                obs = _strip_shell_command_echo(obs)
                obs = _smart_truncate(obs)
            elif len(obs) > _OBS_LIMIT:
                obs = obs[:_OBS_LIMIT]
            tc.observation = obs


def _apply_abort_info(structure: ExecutionStructure, trajectory_md: str) -> ExecutionStructure:
    """Detect [PIPELINE ABORT] markers in the raw trajectory and set run_aborted
    fields on the matching ProcedureNode and ToolCallRecord programmatically.

    The abort message written by run_skill_agent.py embeds the tool name
    as  tool='<name>'  so we extract it directly from the message rather than
    relying on positional matching.
    """
    if "[PIPELINE ABORT]" not in trajectory_md:
        return structure

    # Extract every abort block from the trajectory: the observation section
    # for an aborted tool call contains the full [ABORT] message on one line.
    abort_pattern = re.compile(
        r"\[ABORT\][^\n]*tool='([^']+)'[^\n]*",
    )
    matches = abort_pattern.findall(trajectory_md)
    if not matches:
        return structure

    aborted_tool_names: set[str] = set(matches)

    # Extract the first full abort message for abort_reason
    first_abort = re.search(r"\[ABORT\][^\n]+", trajectory_md)
    abort_msg = first_abort.group(0) if first_abort else "[PIPELINE ABORT] run terminated"

    raw = structure.model_dump()
    patched = False
    for node in raw["nodes"]:
        for tc in node.get("tool_calls", []):
            if tc["tool"] in aborted_tool_names and not tc.get("aborted_run"):
                tc["aborted_run"] = True
                tc["outcome"] = "error"
                if not tc.get("output_summary") or "[ABORT]" not in tc.get("output_summary", ""):
                    tc["output_summary"] = abort_msg
                node["run_aborted"] = True
                node["abort_reason"] = abort_msg
                node["outcome"] = "error"
                patched = True
    if patched:
        raw["outcome"] = "error"

    return ExecutionStructure.model_validate(raw)


# ──────────────────────────────────────────────────────────────────────────────
# TrajectoryAbstractor
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryAbstractor:
    """Convert a raw trajectory markdown or json string into an ExecutionStructure.

    Drives an agentic loop: the agent can read files, run shell commands for
    deeper inspection, call validate_draft() to self-correct, then calls
    submit_structure() to finalise.  The public abstract() method is
    synchronous and unchanged from the previous implementation.
    """

    def __init__(self, model: str = "gpt-5.4", project_root: Path | None = None,
                 max_turns: int = 30, model_kwargs: dict | None = None):
        self.model = model
        self.model_kwargs = model_kwargs
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.max_turns = max_turns
        # Accumulated usage across all abstract() / abstract_pair() calls.
        # Each entry is an agents.Usage object (or None if unavailable).
        self._stream_results: list = []

    def abstract(
        self,
        trajectory_md: str,
        agent_id: str,
        task: str,
        output_path: Optional[Path] = None,
        trajectory_path: Optional[Path] = None,
        reference_structure_json: Optional[str] = None,
        reference_trajectory_md: Optional[str] = None,
        raw_log_path: Optional[Path] = None,
    ) -> ExecutionStructure:
        """Run the abstractor agent; validate and return the ExecutionStructure.

        If output_path is given, write the JSON artefact to that file.
        If trajectory_path is given, write a human-readable markdown trajectory
        (agent thoughts, tool calls, observations) to that file.
        If reference_structure_json and/or reference_trajectory_md are given,
        they are included in the prompt as a label-alignment guide — the agent
        is instructed to use the same tier2 labels for equivalent steps.
        Providing both gives the richest context: the structure for label lookup
        and the raw trajectory for deeper procedural understanding.
        Raises ValueError if the agent fails to submit a valid structure.
        """
        has_ref = bool(reference_structure_json or reference_trajectory_md)
        logger.info(
            f"Abstracting trajectory for agent '{agent_id}' "
            f"using model '{self.model}'"
            + (" (with strong reference)" if has_ref else "")
            + "..."
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run(
            trajectory_md, agent_id, task,
            output_path, trajectory_path,
            reference_structure_json, reference_trajectory_md,
            raw_log_path,
        )
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def abstract_pair(
        self,
        strong_md: str,
        weak_md: str,
        strong_id: str,
        weak_id: str,
        task: str,
        strong_output_path: Optional[Path] = None,
        weak_output_path: Optional[Path] = None,
        trajectory_path: Optional[Path] = None,
        raw_log_path: Optional[Path] = None,
    ) -> tuple[ExecutionStructure, ExecutionStructure]:
        """Abstract strong and weak trajectories together in a single agent run.

        The agent receives both trajectories at once and is instructed to use
        consistent tier1/tier2 labels across both, making gap analysis more
        semantically aligned than two independent runs.

        Returns (strong_structure, weak_structure).
        """
        logger.info(
            f"Joint abstraction of '{strong_id}' and '{weak_id}' "
            f"using model '{self.model}'..."
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_pair(
            strong_md, weak_md, strong_id, weak_id, task,
            strong_output_path, weak_output_path, trajectory_path,
            raw_log_path,
        )
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    # ── internal async implementation ────────────────────────────────────────

    async def _run(
        self,
        trajectory_md: str,
        agent_id: str,
        task: str,
        output_path: Optional[Path],
        trajectory_path: Optional[Path],
        reference_structure_json: Optional[str] = None,
        reference_trajectory_md: Optional[str] = None,
        raw_log_path: Optional[Path] = None,
    ) -> ExecutionStructure:
        from trajectory_parser import parse_tool_calls, inject_code_snippets, inject_observations
        _traj_calls = parse_tool_calls(trajectory_md)

        store = _StructureStore()
        agent = self._build_agent(store, agent_id, task)
        user_msg = _build_user_message(
            trajectory_md, agent_id, task,
            reference_structure_json, reference_trajectory_md,
        )

        # ── open trajectory / raw-log files if requested ──────────────────
        traj_f    = None
        raw_log_f = None
        if trajectory_path is not None:
            traj_path = Path(trajectory_path)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_f = traj_path.open("w", encoding="utf-8")
            traj_f.write(
                f"# Trajectory Abstractor — {agent_id}\n\n"
                f"**model**: {self.model}  \n"
                f"**agent_id**: {agent_id}\n\n"
            )
        if raw_log_path is not None:
            rl = Path(raw_log_path)
            rl.parent.mkdir(parents=True, exist_ok=True)
            raw_log_f = rl.open("w", encoding="utf-8")

        # ── streaming buffers ──────────────────────────────────────────────
        thought_buf: list[str] = []
        tool_name: str | None = None
        tool_args: list[str] = []

        def _write(text: str) -> None:
            if traj_f:
                traj_f.write(text)
                traj_f.flush()

        def _flush_thought() -> None:
            nonlocal thought_buf
            text = "".join(thought_buf).strip()
            if text:
                _write(f"\n### 🤖 Agent\n\n{text}\n")
            thought_buf = []

        def _flush_tool() -> None:
            nonlocal tool_name, tool_args
            if not tool_name:
                return
            args_str = "".join(tool_args)
            _write(f"\n### 🛠 Tool Call: `{tool_name}`\n```json\n{args_str}\n```\n")
            tool_name = None
            tool_args = []

        def _extract_obs(event) -> str:
            for attr in ("output", "content"):
                try:
                    v = getattr(event, attr, None)
                    if v:
                        return str(v)
                except Exception:
                    pass
            try:
                item = getattr(event, "item", None)
                raw = getattr(item, "raw_item", None) if item else None
                if isinstance(raw, dict):
                    return str(raw.get("output") or raw.get("content") or "")
                if raw:
                    return str(getattr(raw, "output", None) or getattr(raw, "content", None) or "")
            except Exception:
                pass
            return ""

        # ── stream ────────────────────────────────────────────────────────
        try:
            stream = Runner.run_streamed(agent, input=user_msg, max_turns=self.max_turns)
            async for event in stream.stream_events():
                if raw_log_f:
                    raw_log_f.write(repr(event) + "\n")
                if traj_f is None:
                    continue

                e_type = _get_event_type(event)
                run_item_name = getattr(event, "name", None) or ""

                if run_item_name == "tool_output" or "tool_output" in e_type:
                    _flush_thought()
                    _flush_tool()
                    obs = _extract_obs(event).strip() or "(empty)"
                    if len(obs) > 2000:
                        obs = obs[:2000] + f"\n... (truncated)"
                    _write(f"\n### 👁 Observation\n```\n{obs}\n```\n")

                elif _looks_like_tool_event(event):
                    t_name, t_args = _extract_tool_call_name_and_args(event)
                    if t_name and t_name != tool_name:
                        _flush_thought()
                        _flush_tool()
                        tool_name = t_name
                    if t_args:
                        joined = "".join(tool_args)
                        if len(t_args) >= len(joined):
                            tool_args = [t_args]
                        else:
                            tool_args.append(t_args)

                else:
                    delta = _extract_text_delta(event)
                    if delta:
                        thought_buf.append(delta)

            _flush_thought()
            _flush_tool()
            self._stream_results.append(stream)

        finally:
            if traj_f:
                traj_f.close()
            if raw_log_f:
                raw_log_f.close()

        # ── validate result ────────────────────────────────────────────────
        if store.structure is None:
            raise ValueError(
                f"TrajectoryAbstractor: agent did not call submit_structure() "
                f"for agent_id='{agent_id}'. Check the model output."
            )

        # Post-process: fill in real code_snippets AND raw observations from
        # the pre-extracted tool calls (LLM tends to self-truncate observations
        # to ~200 chars with an ellipsis, hiding the signal-bearing tail like
        # SyntaxError / Traceback / exit_code from downstream gap analysis).
        raw_dict = store.structure.model_dump()
        raw_dict = inject_observations(raw_dict, _traj_calls)
        raw_dict = inject_code_snippets(raw_dict, _traj_calls)
        from execution_structure import ExecutionStructure as _ES
        structure = _ES.model_validate(raw_dict)
        _truncate_observations(structure)
        structure = _apply_abort_info(structure, trajectory_md)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                structure.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(f"Written ExecutionStructure to {output_path}")

        return structure

    async def _run_pair(
        self,
        strong_md: str,
        weak_md: str,
        strong_id: str,
        weak_id: str,
        task: str,
        strong_output_path: Optional[Path],
        weak_output_path: Optional[Path],
        trajectory_path: Optional[Path],
        raw_log_path: Optional[Path] = None,
    ) -> tuple[ExecutionStructure, ExecutionStructure]:
        from trajectory_parser import parse_tool_calls, inject_code_snippets, inject_observations
        _strong_calls = parse_tool_calls(strong_md)
        _weak_calls   = parse_tool_calls(weak_md)

        pair = _PairStore()
        agent = self._build_pair_agent(pair, strong_id, weak_id, task)
        user_msg = _build_joint_user_message(
            strong_md, weak_md, strong_id, weak_id, task,
            strong_tool_calls=_strong_calls,
            weak_tool_calls=_weak_calls,
        )

        traj_f    = None
        raw_log_f = None
        if trajectory_path is not None:
            traj_path = Path(trajectory_path)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_f = traj_path.open("w", encoding="utf-8")
            traj_f.write(
                f"# Trajectory Abstractor — Joint ({strong_id} + {weak_id})\n\n"
                f"**model**: {self.model}  \n"
                f"**mode**: joint\n\n"
            )
        if raw_log_path is not None:
            rl = Path(raw_log_path)
            rl.parent.mkdir(parents=True, exist_ok=True)
            raw_log_f = rl.open("w", encoding="utf-8")

        thought_buf: list[str] = []
        tool_name: str | None = None
        tool_args: list[str] = []

        def _write(text: str) -> None:
            if traj_f:
                traj_f.write(text)
                traj_f.flush()

        def _flush_thought() -> None:
            nonlocal thought_buf
            text = "".join(thought_buf).strip()
            if text:
                _write(f"\n### 🤖 Agent\n\n{text}\n")
            thought_buf = []

        def _flush_tool() -> None:
            nonlocal tool_name, tool_args
            if not tool_name:
                return
            args_str = "".join(tool_args)
            _write(f"\n### 🛠 Tool Call: `{tool_name}`\n```json\n{args_str}\n```\n")
            tool_name = None
            tool_args = []

        def _extract_obs(event) -> str:
            for attr in ("output", "content"):
                try:
                    v = getattr(event, attr, None)
                    if v:
                        return str(v)
                except Exception:
                    pass
            try:
                item = getattr(event, "item", None)
                raw = getattr(item, "raw_item", None) if item else None
                if isinstance(raw, dict):
                    return str(raw.get("output") or raw.get("content") or "")
                if raw:
                    return str(getattr(raw, "output", None) or getattr(raw, "content", None) or "")
            except Exception:
                pass
            return ""

        try:
            stream = Runner.run_streamed(agent, input=user_msg, max_turns=self.max_turns)
            async for event in stream.stream_events():
                if raw_log_f:
                    raw_log_f.write(repr(event) + "\n")
                if traj_f is None:
                    continue
                e_type = _get_event_type(event)
                run_item_name = getattr(event, "name", None) or ""

                if run_item_name == "tool_output" or "tool_output" in e_type:
                    _flush_thought()
                    _flush_tool()
                    obs = _extract_obs(event).strip() or "(empty)"
                    if len(obs) > 2000:
                        obs = obs[:2000] + "\n... (truncated)"
                    _write(f"\n### 👁 Observation\n```\n{obs}\n```\n")
                elif _looks_like_tool_event(event):
                    t_name, t_args = _extract_tool_call_name_and_args(event)
                    if t_name and t_name != tool_name:
                        _flush_thought()
                        _flush_tool()
                        tool_name = t_name
                    if t_args:
                        joined = "".join(tool_args)
                        if len(t_args) >= len(joined):
                            tool_args = [t_args]
                        else:
                            tool_args.append(t_args)
                else:
                    delta = _extract_text_delta(event)
                    if delta:
                        thought_buf.append(delta)

            _flush_thought()
            _flush_tool()
            self._stream_results.append(stream)
        finally:
            if traj_f:
                traj_f.close()
            if raw_log_f:
                raw_log_f.close()

        if pair.strong is None:
            raise ValueError(
                f"Joint abstractor: agent did not call submit_strong_structure() "
                f"for '{strong_id}'."
            )
        if pair.weak is None:
            raise ValueError(
                f"Joint abstractor: agent did not call submit_weak_structure() "
                f"for '{weak_id}'."
            )

        # Post-process: fill in real code_snippets AND raw observations from
        # pre-extracted tool calls (see solo-path comment above for rationale).
        from execution_structure import ExecutionStructure as _ES
        strong_dict = inject_observations(pair.strong.model_dump(), _strong_calls)
        weak_dict   = inject_observations(pair.weak.model_dump(),   _weak_calls)
        strong_dict = inject_code_snippets(strong_dict, _strong_calls)
        weak_dict   = inject_code_snippets(weak_dict,   _weak_calls)
        strong_out_struct = _ES.model_validate(strong_dict)
        weak_out_struct   = _ES.model_validate(weak_dict)
        _truncate_observations(strong_out_struct)
        _truncate_observations(weak_out_struct)
        strong_out_struct = _apply_abort_info(strong_out_struct, strong_md)
        weak_out_struct   = _apply_abort_info(weak_out_struct,   weak_md)

        for structure, out_path in [
            (strong_out_struct, strong_output_path),
            (weak_out_struct,   weak_output_path),
        ]:
            if out_path is not None:
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(structure.model_dump_json(indent=2), encoding="utf-8")
                logger.info(f"Written ExecutionStructure to {out_path}")

        return strong_out_struct, weak_out_struct

    def _build_agent(
        self, store: _StructureStore, agent_id: str, task: str
    ) -> Agent:
        root = self.project_root

        # ── read_file ─────────────────────────────────────────────────────────
        @function_tool
        def read_file(file_path: str) -> str:
            """Read a text file for deeper inspection.
            Useful for re-reading the trajectory or examining output files
            produced by the agent being abstracted.
            Path may be absolute or relative to project root."""
            p = Path(file_path)
            if not p.is_absolute():
                p = root / file_path
            p = p.resolve()
            if not str(p).startswith(str(root)):
                return f"Error: '{file_path}' is outside the project directory."
            if not p.exists():
                return f"Error: file '{file_path}' not found."
            try:
                text = p.read_text(encoding="utf-8")
                # Cap at 8 000 chars to avoid overwhelming the context
                if len(text) > 8000:
                    return text[:8000] + f"\n... (truncated — {len(text)} chars total)"
                return text
            except Exception as exc:
                return f"Error reading file: {exc}"

        # ── shell ─────────────────────────────────────────────────────────────
        @function_tool
        def shell(commands: list[str], timeout_ms: int = 15000) -> str:
            """Run shell commands in the project directory.
            Useful for inspecting output files (e.g. reading a CSV header,
            counting rows) to better understand what the agent produced."""
            timeout_sec = timeout_ms / 1000.0
            outputs: list[str] = []
            for cmd in commands:
                try:
                    proc = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True,
                        timeout=timeout_sec, cwd=str(root),
                        env={**os.environ},
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

        # ── validate_draft ────────────────────────────────────────────────────
        @function_tool
        def validate_draft(json_str: str) -> str:
            """Validate a draft ExecutionStructure JSON against the Pydantic schema.
            Returns 'Valid: N nodes' on success, or a description of the error.
            Use this to catch schema mistakes before calling submit_structure()."""
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as exc:
                return f"JSON syntax error: {exc}"

            data.setdefault("agent_id", agent_id)
            data.setdefault("task", task)
            if "nodes" in data:
                data["total_steps"] = len(data["nodes"])

            try:
                s = ExecutionStructure.model_validate(data)
                return (
                    f"Valid: {s.total_steps} nodes, outcome={s.outcome}, "
                    f"tier1 categories={sorted({n.category_tier1 for n in s.nodes})}"
                )
            except Exception as exc:
                return f"Validation error: {exc}"

        # ── submit_structure ──────────────────────────────────────────────────
        @function_tool
        def submit_structure(json_str: str) -> str:
            """Submit the final ExecutionStructure JSON.
            The JSON is validated and stored as the abstraction result.
            Call this once when you are satisfied with the structure."""
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as exc:
                return f"JSON syntax error — not submitted: {exc}"

            data.setdefault("agent_id", agent_id)
            data.setdefault("task", task)
            if "nodes" in data:
                data["total_steps"] = len(data["nodes"])

            try:
                structure = ExecutionStructure.model_validate(data)
            except Exception as exc:
                return (
                    f"Validation error — not submitted: {exc}\n"
                    f"Fix the error and call submit_structure() again."
                )

            store.structure = structure
            return (
                f"Submitted: {structure.total_steps} nodes, "
                f"outcome={structure.outcome}, "
                f"tier1={sorted({n.category_tier1 for n in structure.nodes})}"
            )

        return Agent(
            name="TrajectoryAbstractor",
            instructions=_ABSTRACTOR_SYSTEM_PROMPT,
            tools=[read_file, shell, validate_draft, submit_structure],
            model=self.model,
            **(self.model_kwargs or {}),
        )

    def _build_pair_agent(
        self, pair: _PairStore, strong_id: str, weak_id: str, task: str
    ) -> Agent:
        root = self.project_root

        # Shared helpers (same as _build_agent) ────────────────────────────────
        @function_tool
        def read_file(file_path: str) -> str:
            """Read a text file. Path may be absolute or relative to project root."""
            p = Path(file_path)
            if not p.is_absolute():
                p = root / file_path
            p = p.resolve()
            if not str(p).startswith(str(root)):
                return f"Error: '{file_path}' is outside the project directory."
            if not p.exists():
                return f"Error: file '{file_path}' not found."
            try:
                text = p.read_text(encoding="utf-8")
                if len(text) > 8000:
                    return text[:8000] + f"\n... (truncated — {len(text)} chars total)"
                return text
            except Exception as exc:
                return f"Error reading file: {exc}"

        @function_tool
        def shell(commands: list[str], timeout_ms: int = 15000) -> str:
            """Run shell commands in the project directory."""
            timeout_sec = timeout_ms / 1000.0
            outputs: list[str] = []
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

        @function_tool
        def validate_draft(agent_id: str, json_str: str) -> str:
            """Validate a draft ExecutionStructure JSON.
            agent_id: either the strong or weak agent's id (for logging only).
            Returns 'Valid: N nodes' or a description of the error."""
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as exc:
                return f"JSON syntax error: {exc}"
            data.setdefault("agent_id", agent_id)
            data.setdefault("task", task)
            if "nodes" in data:
                data["total_steps"] = len(data["nodes"])
            try:
                s = ExecutionStructure.model_validate(data)
                return (
                    f"Valid ({agent_id}): {s.total_steps} nodes, outcome={s.outcome}, "
                    f"tier1={sorted({n.category_tier1 for n in s.nodes})}"
                )
            except Exception as exc:
                return f"Validation error ({agent_id}): {exc}"

        # ── per-agent submit tools ────────────────────────────────────────────
        def _make_submit(label: str, target_id: str):
            def _submit(json_str: str) -> str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    return f"JSON syntax error — not submitted: {exc}"
                data.setdefault("agent_id", target_id)
                data.setdefault("task", task)
                if "nodes" in data:
                    data["total_steps"] = len(data["nodes"])
                try:
                    structure = ExecutionStructure.model_validate(data)
                except Exception as exc:
                    return (
                        f"Validation error — not submitted: {exc}\n"
                        f"Fix the error and call {label}() again."
                    )
                if label == "submit_strong_structure":
                    pair.strong = structure
                else:
                    pair.weak = structure
                return (
                    f"Submitted ({target_id}): {structure.total_steps} nodes, "
                    f"outcome={structure.outcome}, "
                    f"tier1={sorted({n.category_tier1 for n in structure.nodes})}"
                )
            _submit.__name__ = label
            _submit.__doc__ = (
                f"Submit the final ExecutionStructure JSON for the "
                f"{'strong' if 'strong' in label else 'weak'} agent ({target_id}).\n"
                f"Call this once when you are satisfied with that agent's structure."
            )
            return function_tool(_submit)

        submit_strong = _make_submit("submit_strong_structure", strong_id)
        submit_weak   = _make_submit("submit_weak_structure",   weak_id)

        return Agent(
            name="TrajectoryAbstractorJoint",
            instructions=_ABSTRACTOR_JOINT_SYSTEM_PROMPT,
            tools=[read_file, shell, validate_draft, submit_strong, submit_weak],
            model=self.model,
            **(self.model_kwargs or {}),
        )
