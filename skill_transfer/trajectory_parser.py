"""
trajectory_parser.py — Extract tool calls from trajectory files (markdown or JSON).

Supports two formats:
  - Markdown  (legacy): ### 🛠 Tool Call / ### 👁 Observation blocks
  - JSON      (v1):     {"format": "trajectory_json_v1", "steps": [...]}
    Each step is {"type": "thought"|"tool_call", ...} with tool calls already
    paired with their observation.

Used by TrajectoryAbstractor to:
1. Pre-extract code_snippets (avoids LLM reproducing raw code → no escape errors)
2. Replace tool call blocks with compact [#N: tool] markers (reduces context size)
3. Post-process LLM output: replace "#N" references with real code_snippets

All public functions auto-detect format — callers need no changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedToolCall:
    idx: int
    tool_name: str
    args_raw: str       # raw JSON string as it appeared in trajectory
    code_snippet: str   # extracted code: commands list for shell, content for write_file, else args_raw
    observation: str    # truncated observation text following this call (may be empty)


# ── regex patterns (markdown format) ─────────────────────────────────────────

# Matches: ### 🛠 Tool Call: `name`\n```[json]\n{...}\n```
_TOOL_CALL_RE = re.compile(
    r'### 🛠 Tool Call: `([^`]+)`\s*\n```[^\n]*\n(.*?)\n```',
    re.DOTALL,
)

# Matches: ### 👁 Observation[...]: \n```[text]\n...\n``` or --- separator
_OBSERVATION_RE = re.compile(
    r'### 👁 Observation[^\n]*\n```[^\n]*\n(.*?)(?:\n```|\n---)',
    re.DOTALL,
)


# ── format detection ──────────────────────────────────────────────────────────

def _is_json_trajectory(text: str) -> bool:
    """Return True if text is a trajectory_json_v1 document."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
        return data.get("format") == "trajectory_json_v1"
    except Exception:
        return False


# ── shared helper ─────────────────────────────────────────────────────────────

def _extract_code_snippet(tool_name: str, args_raw: str) -> str:
    """Extract the most meaningful code/command from a tool call's raw JSON args."""
    try:
        args = json.loads(args_raw)
    except Exception:
        return args_raw

    if tool_name == "shell":
        commands = args.get("commands", [])
        if isinstance(commands, list):
            return "\n".join(str(c) for c in commands)
        return str(commands)

    if tool_name == "write_file":
        content = args.get("content", "")
        return content if isinstance(content, str) else str(content)

    # For read_file, glob_files, activate_skill, grep_files, etc.
    return args_raw


# ── JSON-format parsers ───────────────────────────────────────────────────────

# Per-tool observation char limits applied at parse time.
# `shell` is special-cased: its observation typically ends with the signal-
# bearing portion (stderr / exit_code), and the trailing portion is what
# downstream gap analysis needs. The "Command: …" prefix can be very long
# but is stripped in a later post-processing pass, after which a smaller
# final cap is applied. So we do NOT truncate shell observations at parse
# time. Other tools keep the legacy 300-char limit.
_TOOL_OBS_LIMIT_DEFAULT = 300
_TOOL_OBS_LIMITS: dict[str, int | None] = {
    "shell": None,         # parse-time: no limit (post-process handles trimming)
}


def _per_tool_limit(tool_name: str, override: int | None = None) -> int | None:
    if override is not None:
        return override
    return _TOOL_OBS_LIMITS.get(tool_name, _TOOL_OBS_LIMIT_DEFAULT)


def _maybe_truncate(observation: str, limit: int | None) -> str:
    if limit is None or len(observation) <= limit:
        return observation
    return observation[:limit] + "…"


def _parse_tool_calls_json(
    json_text: str,
    max_obs_chars: int | None = None,
) -> list[ParsedToolCall]:
    """Parse a trajectory_json_v1 document into ParsedToolCall objects.

    Observation truncation is tool-aware (see _TOOL_OBS_LIMITS). Passing
    `max_obs_chars` overrides the per-tool limit uniformly for callers that
    need a strict global cap (e.g. compact-trajectory rendering).
    """
    data = json.loads(json_text)
    steps = data.get("steps", [])
    calls: list[ParsedToolCall] = []
    call_idx = 0
    for step in steps:
        if step.get("type") != "tool_call":
            continue
        tool_name = step.get("tool", "")
        args = step.get("args", {})
        args_raw = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
        observation = str(step.get("observation") or "").strip()
        observation = _maybe_truncate(observation, _per_tool_limit(tool_name, max_obs_chars))
        calls.append(ParsedToolCall(
            idx=call_idx,
            tool_name=tool_name,
            args_raw=args_raw,
            code_snippet=_extract_code_snippet(tool_name, args_raw),
            observation=observation,
        ))
        call_idx += 1
    return calls


def _compact_trajectory_json(
    json_text: str,
    tool_calls: list[ParsedToolCall],
    max_obs_chars: int = 200,
    max_thought_chars: int = 300,
) -> str:
    """
    Produce a compact text representation of a JSON trajectory for the LLM prompt.

    Thoughts are preserved (truncated); tool calls become [#N: tool_name] markers
    with a short observation suffix — the same style as compact_trajectory for markdown.

    Example output:
        [Thought] I need to activate the xlsx skill first.
        [#0: activate_skill] → {"name": "xlsx"}
        [Thought] Now I'll read the instruction to understand the task.
        [#1: read_file] → # Task: Extract table from sheet "Sales"…
        [#2: shell] → Found 3 sheets: Sales, Summary, Raw…
    """
    data = json.loads(json_text)
    steps = data.get("steps", [])
    call_map = {tc.idx: tc for tc in tool_calls}
    call_counter = 0
    lines: list[str] = []

    for step in steps:
        step_type = step.get("type")
        if step_type == "thought":
            content = step.get("content", "").strip()
            if content:
                if len(content) > max_thought_chars:
                    content = content[:max_thought_chars] + "…"
                # Collapse newlines for compactness
                lines.append("[Thought] " + content.replace("\n", " "))
        elif step_type == "tool_call":
            tc = call_map.get(call_counter)
            if tc:
                obs = tc.observation or ""
                if len(obs) > max_obs_chars:
                    obs = obs[:max_obs_chars] + "…"
                obs_str = " → " + obs.replace("\n", " ") if obs else ""
                lines.append(f"[#{tc.idx}: {tc.tool_name}]{obs_str}")
            call_counter += 1

    return "\n".join(lines)


# ── public API (auto-detecting) ───────────────────────────────────────────────

def parse_tool_calls(
    trajectory: str,
    max_obs_chars: int | None = None,
) -> list[ParsedToolCall]:
    """Parse a trajectory (markdown or JSON) and return all tool calls in order.

    Auto-detects format: if the text is a trajectory_json_v1 document the JSON
    parser is used; otherwise the legacy markdown regex parser is used.

    Observation truncation is tool-aware (see _TOOL_OBS_LIMITS). Pass
    `max_obs_chars` to override the per-tool limit with a strict global cap.
    """
    if _is_json_trajectory(trajectory):
        return _parse_tool_calls_json(trajectory, max_obs_chars=max_obs_chars)

    # ── markdown path (legacy) ────────────────────────────────────────────────
    tool_matches = list(_TOOL_CALL_RE.finditer(trajectory))
    obs_matches  = list(_OBSERVATION_RE.finditer(trajectory))

    calls: list[ParsedToolCall] = []
    for i, m in enumerate(tool_matches):
        tool_name = m.group(1)
        args_raw  = m.group(2).strip()
        tc_end    = m.end()

        # Observation window: between end of this tool call and start of next
        window_end = tool_matches[i + 1].start() if i + 1 < len(tool_matches) else len(trajectory)
        relevant_obs = [
            o.group(1).strip()
            for o in obs_matches
            if tc_end <= o.start() < window_end
        ]
        observation = " | ".join(relevant_obs)
        observation = _maybe_truncate(observation, _per_tool_limit(tool_name, max_obs_chars))

        calls.append(ParsedToolCall(
            idx=i,
            tool_name=tool_name,
            args_raw=args_raw,
            code_snippet=_extract_code_snippet(tool_name, args_raw),
            observation=observation,
        ))

    return calls


def compact_trajectory(
    trajectory: str,
    tool_calls: list[ParsedToolCall],
    max_obs_chars: int = 200,
) -> str:
    """Return a compact version of the trajectory for use in the abstractor prompt.

    Auto-detects format:
      - JSON:     produces [Thought] / [#N: tool] lines (no raw args in output)
      - Markdown: replaces tool call arg blocks with [#N: tool_name] markers,
                  preserving agent thought text verbatim.

    Both outputs are suitable for feeding to the TrajectoryAbstractor LLM.
    """
    if _is_json_trajectory(trajectory):
        return _compact_trajectory_json(trajectory, tool_calls, max_obs_chars=max_obs_chars)

    # ── markdown path (legacy) ────────────────────────────────────────────────
    result = trajectory
    tool_matches = list(_TOOL_CALL_RE.finditer(trajectory))
    obs_matches  = list(_OBSERVATION_RE.finditer(trajectory))

    replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

    for i, m in enumerate(tool_matches):
        tc = tool_calls[i]
        window_end = tool_matches[i + 1].start() if i + 1 < len(tool_matches) else len(trajectory)

        # Find observation block(s) that immediately follow this tool call
        obs_in_window = [o for o in obs_matches if m.end() <= o.start() < window_end]

        obs_text = ""
        obs_end  = m.end()
        if obs_in_window:
            obs_raw = obs_in_window[-1].group(1).strip()
            if len(obs_raw) > max_obs_chars:
                obs_raw = obs_raw[:max_obs_chars] + "…"
            obs_text = " " + obs_raw.replace("\n", " ")
            obs_end  = obs_in_window[-1].end()

        marker = f"[#{tc.idx}: {tc.tool_name}]{obs_text}"
        replacements.append((m.start(), obs_end, marker))

    for start, end, text in reversed(replacements):
        result = result[:start] + text + result[end:]

    return result.strip()


def inject_code_snippets(
    structure_dict: dict,
    tool_calls: list[ParsedToolCall],
) -> dict:
    """
    Post-process an ExecutionStructure dict produced by the LLM.
    Replace code_snippet values of the form "#N" with the actual code
    from the pre-extracted tool_calls list.
    """
    ref_pattern = re.compile(r'^#(\d+)$')
    lookup = {tc.idx: tc.code_snippet for tc in tool_calls}

    for node in structure_dict.get("nodes", []):
        for tc_rec in node.get("tool_calls", []):
            snippet = tc_rec.get("code_snippet", "")
            if isinstance(snippet, str):
                m = ref_pattern.match(snippet.strip())
                if m:
                    idx = int(m.group(1))
                    tc_rec["code_snippet"] = lookup.get(idx, snippet)

    return structure_dict


def inject_observations(
    structure_dict: dict,
    tool_calls: list[ParsedToolCall],
) -> dict:
    """
    Post-process an ExecutionStructure dict produced by the LLM.
    Replace each ToolCallRecord.observation with the raw observation parsed
    directly from the trajectory.

    Why: the abstractor LLM tends to self-truncate observations to ~200
    chars with an ellipsis, losing the signal-bearing tail (stderr,
    Traceback, #VALUE!, exit_code). Matching back to the raw parsed call
    by position recovers the full observation. Downstream post-processing
    in trajectory_abstractor (_truncate_observations) then strips the
    redundant "Command: …" prefix on shell calls and applies the final cap.

    Matching strategy: walk the structure's tool_calls in document order
    and pair with the parsed calls 1:1, but only overwrite when the tool
    names match — protects against rare structural misalignment.
    """
    parsed = list(tool_calls)
    pos = 0
    for node in structure_dict.get("nodes", []):
        for tc_rec in node.get("tool_calls", []):
            if pos < len(parsed) and tc_rec.get("tool") == parsed[pos].tool_name:
                tc_rec["observation"] = parsed[pos].observation
            pos += 1
    return structure_dict


def build_reference_table(tool_calls: list[ParsedToolCall], max_snippet: int = 80) -> str:
    """
    Build a compact numbered reference table to include in the abstractor prompt.

    Example output:
        #0  activate_skill  {"name":"xlsx"}
        #1  read_file       {"file_path":"...INSTRUCTION.md"}
        #2  shell           python3 -c "import pandas as pd..."
        ...
    """
    lines = ["## Tool Call Reference (use #N for code_snippet)\n"]
    for tc in tool_calls:
        snippet = tc.code_snippet.replace("\n", " ").strip()
        if len(snippet) > max_snippet:
            snippet = snippet[:max_snippet] + "…"
        obs = tc.observation.replace("\n", " ")[:60] + "…" if len(tc.observation) > 60 else tc.observation
        obs_str = f"  → {obs}" if obs else ""
        lines.append(f"  #{tc.idx:<3} {tc.tool_name:<20} {snippet}{obs_str}")
    return "\n".join(lines)
