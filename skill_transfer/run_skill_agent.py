"""Run a SkillAgent task from a test-case folder (streamed mode).

This script is built on `run_skill_agent_stream.py`, but is specialized for
running a *case directory* that contains an `INSTRUCTION.md` and one (or more)
PDF inputs.

Example:
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

python3.11 run_skill_agent.py \
--case-dir cases/example \
--skills-dir skills \
--project-root . \
--model gpt-5.4-mini \
--mode messages \
--extra "While executing, continuously report in brief bullet points: the current step, which tool/skill you are about to call, and which files have been produced so far. Do not only summarize at the end." \
--raw-log-file new_gpt5-4-mini_raw_log.txt \
--out-dir new_extracted_tables_gpt5-4-mini \
--name-prefix new_extracted_table_ \
--dump-shell-scripts new_extracted_scripts_gpt5-4-mini \
--trajectory-file new_gpt5-4-mini_trajectory.md

python3.11 run_skill_agent.py \
--case-dir cases/example \
--skills-dir skills \
--project-root . \
--model gpt-5.4 \
--mode messages \
--extra "While executing, continuously report in brief bullet points: the current step, which tool/skill you are about to call, and which files have been produced so far. Do not only summarize at the end." \
--raw-log-file gpt5-4_raw_log.txt \
--out-dir outputs_gpt5-4 \
--name-prefix output_ \
--dump-shell-scripts extracted_scripts_gpt5-4 \
--trajectory-file gpt5-4_trajectory.md \
--max-turns 20

Modes:
  - raw:   print every stream event
  - tools: print only tool-related events
    - messages: print only non-tool text deltas (agent-visible progress)
    - skills: print only activate_skill calls (which skills got activated)
  - final: print only the final aggregated text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from openai_setup import setup_openai
setup_openai("skill-agent")

from skill_agent import SkillAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repository root
DEFAULT_CASE_DIR = PROJECT_ROOT / "cases" / "example"
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"


def _format_event(event) -> str:
    try:
        event_type = getattr(event, "type", None) or getattr(event, "event", None)
        if event_type:
            return f"[{event_type}] {event}"
    except Exception:
        pass

    try:
        if isinstance(event, dict):
            event_type = event.get("type") or event.get("event")
            if event_type:
                return f"[{event_type}] {event}"
    except Exception:
        pass

    return repr(event)


def _get_event_type(event) -> str:
    try:
        t = getattr(event, "type", None) or getattr(event, "event", None)
        if isinstance(t, str):
            return t
    except Exception:
        pass
    if isinstance(event, dict):
        t = event.get("type") or event.get("event")
        if isinstance(t, str):
            return t
    return ""


def _looks_like_tool_event(event) -> bool:
    """Heuristic tool-event detector across SDK versions."""
    try:
        run_item_name = getattr(event, "name", None)
        if run_item_name in ("tool_called", "tool_output"):
            return True
    except Exception:
        pass

    if isinstance(event, dict):
        if event.get("name") in ("tool_called", "tool_output"):
            return True

    t = _get_event_type(event).lower()
    if "tool_call" in t or "tool_call_output" in t:
        return True

    try:
        item = getattr(event, "item", None)
        if item is not None:
            item_cls = type(item).__name__.lower()
            if "toolcall" in item_cls or "tool_call" in item_cls:
                return True
    except Exception:
        pass

    return False


def _extract_text_from_raw_item(raw_item) -> str:
    if raw_item is None:
        return ""

    if isinstance(raw_item, dict):
        for key in ("text", "content", "output_text"):
            v = raw_item.get(key)
            if isinstance(v, str) and v:
                return v

        content = raw_item.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                    continue
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
            return "".join(parts)

        return ""

    try:
        v = getattr(raw_item, "text", None)
        if isinstance(v, str) and v:
            return v
    except Exception:
        pass

    try:
        content = getattr(raw_item, "content", None)
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                    continue
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
                    continue
                try:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t:
                        parts.append(t)
                except Exception:
                    pass
            return "".join(parts)
    except Exception:
        pass

    return ""


def _extract_text_delta(event) -> str:
    if isinstance(event, dict):
        for key in ("delta", "text", "content", "output_text"):
            v = event.get(key)
            if isinstance(v, str) and v:
                return v

        msg = event.get("message")
        if isinstance(msg, dict):
            v = msg.get("content") or msg.get("text")
            if isinstance(v, str) and v:
                return v

        item = event.get("item")
        if isinstance(item, dict):
            raw_item = item.get("raw_item")
            extracted = _extract_text_from_raw_item(raw_item)
            if extracted:
                return extracted
        return ""

    for attr in ("delta", "text", "content", "output_text"):
        try:
            v = getattr(event, attr, None)
            if isinstance(v, str) and v:
                return v
        except Exception:
            pass

    try:
        item = getattr(event, "item", None)
        raw_item = getattr(item, "raw_item", None) if item is not None else None
        extracted = _extract_text_from_raw_item(raw_item)
        if extracted:
            return extracted
    except Exception:
        pass

    return ""


def _extract_tool_call_name_and_args(event) -> tuple[str, str]:
    """Best-effort extraction of (tool_name, tool_args) from stream events."""
    raw_item = None

    # Dict event
    if isinstance(event, dict):
        item = event.get("item")
        if isinstance(item, dict):
            raw_item = item.get("raw_item")
        else:
            raw_item = event.get("raw_item")
    else:
        try:
            item = getattr(event, "item", None)
            raw_item = getattr(item, "raw_item", None) if item is not None else None
        except Exception:
            raw_item = None

    if raw_item is None:
        return "", ""

    # Dict raw_item
    if isinstance(raw_item, dict):
        # Common shapes:
        # {"name": "activate_skill", "arguments": "{...}"}
        # {"function": {"name": "...", "arguments": "..."}}
        fn = raw_item.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            args = fn.get("arguments")
            return (name or ""), (args or "")

        name = raw_item.get("name") or raw_item.get("tool_name")
        args = raw_item.get("arguments") or raw_item.get("args")
        if isinstance(args, dict):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = str(args)
        return (name or ""), (args or "")

    # Object raw_item
    try:
        fn = getattr(raw_item, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None) or ""
            args = getattr(fn, "arguments", None) or ""
            return name, args
    except Exception:
        pass

    try:
        name = getattr(raw_item, "name", None) or getattr(raw_item, "tool_name", None) or ""
        args = getattr(raw_item, "arguments", None) or getattr(raw_item, "args", None) or ""
        if isinstance(args, dict):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = str(args)
        return name, args
    except Exception:
        return "", ""


def _extract_tool_call_id(event) -> str:
    """Extract the unique call ID from a tool-call stream event.

    Used by the gap-agent trajectory recorder to distinguish parallel calls
    that share the same tool name (e.g. 14 concurrent write_case_gap_report).
    Returns "" when no ID is present (safe to ignore — falls back to name comparison).
    """
    raw_item = None
    if isinstance(event, dict):
        item = event.get("item")
        raw_item = item.get("raw_item") if isinstance(item, dict) else None
    else:
        try:
            item = getattr(event, "item", None)
            raw_item = getattr(item, "raw_item", None) if item is not None else None
        except Exception:
            raw_item = None

    if raw_item is None:
        return ""
    if isinstance(raw_item, dict):
        return str(raw_item.get("id") or raw_item.get("call_id") or "")
    try:
        return str(getattr(raw_item, "id", None) or getattr(raw_item, "call_id", None) or "")
    except Exception:
        return ""


def _extract_skill_name_from_args(args_text: str) -> str:
    if not args_text:
        return ""
    s = args_text.strip()

    # JSON object string
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                v = obj.get("skill_name") or obj.get("name")
                return v if isinstance(v, str) else ""
        except Exception:
            return ""

    return ""


def _extract_skill_reason_from_args(args_text: str) -> str:
    if not args_text:
        return ""
    s = args_text.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                v = obj.get("reason", "")
                return v if isinstance(v, str) else ""
        except Exception:
            return ""
    return ""


def _try_parse_json(args_text: str):
    if not args_text:
        return None
    s = args_text.strip()
    if not s:
        return None
    if not s.startswith("{"):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


_PY_HEREDOC_RE = re.compile(
    r"^\s*(python3?|python)\s+-\s+<<\s*'?([A-Za-z_][A-Za-z0-9_]*)'?\s*\n(?P<body>[\s\S]*?)\n\2\s*$",
    re.MULTILINE,
)


# Match commands that write a heredoc into a .py file, e.g.:
#   cat > /tmp/x.py <<'PY'
#   ...
#   PY
_WRITE_HEREDOC_TO_FILE_RE_1 = re.compile(
    r"^\s*(?:cat|tee)\b[\s\S]*?>\s*(?P<path>[^\s]+\.py)\s*<<\s*'?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'?\s*\n"
    r"(?P<body>[\s\S]*?)\n(?P=tag)\s*$",
    re.MULTILINE,
)


# Or:
#   cat <<'PY' > /tmp/x.py
#   ...
#   PY
_WRITE_HEREDOC_TO_FILE_RE_2 = re.compile(
    r"^\s*(?:cat|tee)\b[\s\S]*?<<\s*'?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'?\s*>\s*(?P<path>[^\s]+\.py)\s*\n"
    r"(?P<body>[\s\S]*?)\n(?P=tag)\s*$",
    re.MULTILINE,
)


# Match python -c snippets. We capture the quoted string body.
_PYTHON_C_RE = re.compile(
    r"\b(python3?|python)\b\s+-c\s+(?P<q>'|\")(?P<body>(?:\\.|(?!\2)[\s\S])*)(?P=q)",
    re.MULTILINE,
)


# Match execution of a .py file, e.g. `python /tmp/x.py` or `python3 ./x.py`.
_PYTHON_FILE_EXEC_RE = re.compile(
    r'\b(python3?|python)\b\s+(?P<path>(?:\./|/)?[^\s"\']+\.py)\b',
    re.MULTILINE,
)


def _extract_python_heredoc_bodies(command: str) -> list[str]:
    """Extract python heredoc bodies from a shell command string."""
    if not command:
        return []

    m = _PY_HEREDOC_RE.match(command)
    if not m:
        return []
    body = m.group("body")
    if not isinstance(body, str):
        return []
    body = body.strip("\n")
    return [body] if body.strip() else []


def _extract_writefile_heredoc(command: str) -> list[tuple[str, str]]:
    """Extract (target_path, body) when a heredoc is written into a .py file."""
    if not command:
        return []
    m = _WRITE_HEREDOC_TO_FILE_RE_1.match(command) or _WRITE_HEREDOC_TO_FILE_RE_2.match(command)
    if not m:
        return []
    target = (m.group("path") or "").strip()
    body = (m.group("body") or "").strip("\n")
    if not target or not body.strip():
        return []
    return [(target, body)]


def _extract_python_c_snippets(command: str) -> list[str]:
    if not command:
        return []
    m = _PYTHON_C_RE.search(command)
    if not m:
        return []
    body = m.group("body")
    if not isinstance(body, str):
        return []
    # Keep it as-is; it may contain escaped newlines, etc.
    return [body] if body.strip() else []


def _extract_python_exec_paths(command: str) -> list[str]:
    if not command:
        return []
    paths: list[str] = []
    for m in _PYTHON_FILE_EXEC_RE.finditer(command):
        p = m.group("path")
        if isinstance(p, str) and p:
            paths.append(p)
    return paths


def _read_text_if_exists(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _extract_shell_commands_from_args(args_text: str) -> list[str]:
    obj = _try_parse_json(args_text)
    if not isinstance(obj, dict):
        return []
    cmds = obj.get("commands")
    if not isinstance(cmds, list):
        return []
    return [c for c in cmds if isinstance(c, str) and c.strip()]


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:80]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_SKIP_SUFFIXES = {".md", ".json", ".txt", ".log", ".sh", ".py"}


def _pick_input_files(case_dir: Path, explicit: str | None) -> list[Path]:
    """Return the list of task input files in case_dir.

    If --input-file is given, use that path directly.
    Otherwise, return all files in case_dir that are not metadata/code files.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return [p]

    return sorted(
        p for p in case_dir.iterdir()
        if p.is_file() and p.suffix.lower() not in _SKIP_SUFFIXES
    )


def _render_trajectory_json(traj_path: Path) -> str:
    """Convert a trajectory JSON into readable text for context injection."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    steps = data.get("steps", [])
    lines: list[str] = []
    for s in steps:
        stype = s.get("type", "")
        if stype == "thought":
            content = (s.get("content") or "").strip()
            if content:
                lines.append(f"[THOUGHT] {content}")
        elif stype == "tool_call":
            tool = s.get("tool", "")
            args = s.get("args") or {}
            obs = (s.get("observation") or "").strip()
            lines.append(f"[TOOL: {tool}] {json.dumps(args, ensure_ascii=False)}")
            if obs:
                lines.append(f"  → {obs}")
    return "\n".join(lines)


def _build_case_query(
    case_dir: Path,
    instruction_text: str,
    input_files: list[Path],
    out_dir: Path,
    name_prefix: str,
    extra: str,
) -> str:
    instruction_text = instruction_text.strip()
    name_prefix = (name_prefix or "").strip()
    extra = (extra or "").strip()

    query = (
        "You are running a specific test case. Follow the instruction exactly.\n\n"
        f"CASE_DIR: {case_dir}\n"
        f"INSTRUCTION_FILE: {case_dir / 'INSTRUCTION.md'}\n"
    )

    if len(input_files) == 1:
        query += f"INPUT_FILE: {input_files[0]}\n"
    elif input_files:
        query += "INPUT_FILES:\n" + "".join(f"  - {f}\n" for f in input_files)

    query += (
        "\nINSTRUCTION:\n"
        f"{instruction_text}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        f"- Always use ABSOLUTE paths in every tool call and in every script you write.\n"
        f"- shell/write_file root (intermediate files and outputs go here): {out_dir}\n"
        f"- read_file/glob_files/grep_files root (input files here): {case_dir}\n"
    )

    if name_prefix:
        query += (
            f"- Use a stable naming pattern for output files "
            f"(e.g. {name_prefix}001, {name_prefix}002, ...)\n"
        )

    if extra:
        query += "\nEXTRA NOTES:\n" + extra + "\n"

    return query


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="messages",
        choices=["raw", "tools", "messages", "skills", "final"],
        help=(
            "raw: print every stream event; tools: print tool-related events only; "
            "messages: print only non-tool text; skills: print only activate_skill calls; "
            "final: print only the final aggregated text"
        ),
    )
    parser.add_argument(
        "--raw-log-file",
        type=str,
        default="",
        help=(
            "Write the full raw stream (one formatted event per line) to this txt log file. "
            "If relative, it is resolved under CASE_DIR."
        ),
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default="",
        help=(
            "Write a clean, human-readable markdown trajectory to this file. "
            "If relative, it is resolved under CASE_DIR."
        ),
    )
    parser.add_argument(
        "--dump-shell-scripts",
        type=str,
        default="",
        help=(
            "If set, extract Python heredoc scripts from `shell` tool calls (e.g. `python - <<'PY' ... PY`) "
            "and save them as .py files under this directory. If relative, it is resolved under CASE_DIR."
        ),
    )
    parser.add_argument(
        "--case-dir",
        type=str,
        default=str(DEFAULT_CASE_DIR),
        help="Case folder containing INSTRUCTION.md and input PDFs",
    )
    parser.add_argument(
        "--instruction-file",
        type=str,
        default="",
        help="Override instruction file path (defaults to CASE_DIR/INSTRUCTION.md)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        dest="input_file",
        help="Explicit path to the task input file. If omitted, all non-metadata files "
             "in CASE_DIR are discovered automatically.",
    )
    parser.add_argument(
        "--pdf",  # legacy alias for --input-file
        type=str,
        default="",
        dest="input_file",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output directory for task results. If relative, resolved under CASE_DIR. "
            "Defaults to CASE_DIR."
        ),
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="",
        help="Optional filename prefix for output files (e.g. 'result_' → result_001, result_002).",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default=(
            "While executing, continuously report in brief bullet points: "
            "the current step, which tool/skill you are about to call, and "
            "which files have been produced so far. Do not only summarize at the end."
        ),
        help="Extra notes appended to the query (optional)",
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        default=str(DEFAULT_SKILLS_DIR),
        help="Directory containing SKILL.md files",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root for tools (read/write/grep/shell/apply_patch)",
    )
    parser.add_argument(
        "--shell-cwd",
        type=str,
        default="",
        help="Working directory for shell commands. Defaults to --project-root when unset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Model id (must be valid in your OpenAI/compatible setup)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agent-loop turns",
    )
    parser.add_argument(
        "--max-input-tokens-per-turn",
        type=int,
        default=0,
        help=(
            "If > 0, abort the run when any single LLM turn's input token count "
            "exceeds this value. Prevents runaway context inflation from large "
            "tool outputs without truncating what the model sees. 0 = disabled."
        ),
    )
    parser.add_argument(
        "--cost-file",
        type=str,
        default="",
        help="If set, write a JSON cost summary for this run to this path.",
    )
    parser.add_argument(
        "--cost-label",
        type=str,
        default="skill_agent",
        help="Label written into the cost JSON (e.g. 'strong_skill_agent' / 'weak_skill_agent').",
    )
    parser.add_argument(
        "--skills-used-file",
        type=str,
        default="",
        help="If set, write a JSON list of skill names activated during this run to this path.",
    )
    parser.add_argument(
        "--require-skill",
        action="store_true",
        default=False,
        help=(
            "Add a MANDATORY rule to the system prompt requiring the agent to call "
            "activate_skill at least once before acting. Useful when the task type "
            "always has a matching skill (e.g. OfficeQA)."
        ),
    )
    parser.add_argument(
        "--multi-skill",
        action="store_true",
        default=False,
        help=(
            "Tell the agent it may (and should) activate more than one skill. "
            "Useful when tasks span multiple domains (e.g. document search + "
            "statistical analysis)."
        ),
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        default=False,
        help=(
            "Clean no-skill control: ignore all skills AND remove the skill "
            "mechanism — no activate_skill tool, no skill catalog/mandate in the "
            "system prompt, and read_file does not refer the agent to a skill for "
            "xlsx/pdf/etc. Use to measure raw-model performance without skill "
            "scaffolding (point --skills-dir anywhere; its skills are ignored)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="Reasoning effort level for the model: 'low', 'medium', or 'high'. "
             "Empty (default) means no reasoning effort is set.",
    )
    parser.add_argument(
        "--ref-trajectory",
        type=str,
        default="",
        help=(
            "Path to a reference trajectory markdown file from a successful prior run "
            "on this exact case. When provided, the trajectory is appended to the query "
            "so the agent can study the approach. Absolute path required."
        ),
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).expanduser().resolve()
    if not case_dir.exists() or not case_dir.is_dir():
        raise NotADirectoryError(f"case dir not found: {case_dir}")

    instruction_path = (
        Path(args.instruction_file).expanduser().resolve()
        if args.instruction_file
        else (case_dir / "INSTRUCTION.md")
    )
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction file not found: {instruction_path}")

    input_files = _pick_input_files(case_dir, args.input_file or None)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = case_dir / out_dir
        out_dir = out_dir.resolve()
    else:
        out_dir = case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction_text = _read_text_file(instruction_path)
    query = _build_case_query(
        case_dir=case_dir,
        instruction_text=instruction_text,
        input_files=input_files,
        out_dir=out_dir,
        name_prefix=args.name_prefix,
        extra=args.extra,
    )

    if args.ref_trajectory:
        ref_traj_path = Path(args.ref_trajectory).expanduser().resolve()
        if ref_traj_path.exists():
            if ref_traj_path.suffix.lower() == ".json":
                ref_traj_text = _render_trajectory_json(ref_traj_path)
            else:
                ref_traj_text = _read_text_file(ref_traj_path)
            if ref_traj_text:
                query += (
                    "\n\nREFERENCE TRAJECTORY (study the approach — FILE PATHS ARE INVALID in your environment):\n"
                    "⚠️ CRITICAL: Every file path that appears in this trajectory belongs to a DIFFERENT\n"
                    "environment and does NOT exist in your current run. NEVER use any path from this\n"
                    "trajectory directly. Always derive your paths from the CASE_DIR, INPUT_FILE, and\n"
                    "shell/write_file root given above.\n\n"
                    "What you SHOULD learn from this trajectory: the overall approach, which tools to\n"
                    "call in which order, the key data-access patterns, and how to verify the output.\n\n"
                    f"{ref_traj_text}\n\n"
                    "--- END OF REFERENCE TRAJECTORY ---\n\n"
                    "⚠️ YOU HAVE NOT DONE ANYTHING YET. The trajectory above is historical reference only.\n"
                    "You must now execute the task from scratch by calling tools. Start with activate_skill,\n"
                    "then discover and use the actual files in your current environment.\n"
                    "Do NOT output a text summary — use tool calls to perform each step.\n"
                )

    _model_kwargs: dict | None = None
    if args.reasoning_effort:
        from agents import ModelSettings  # noqa: PLC0415
        _model_kwargs = {"model_settings": ModelSettings(reasoning={"effort": args.reasoning_effort})}

    agent = SkillAgent(
        skills_dir=args.skills_dir,
        model=args.model,
        project_root=args.project_root,
        shell_cwd=args.shell_cwd or args.out_dir or args.project_root,
        max_turns=args.max_turns,
        require_skill=args.require_skill,
        multi_skill=args.multi_skill,
        no_skill=args.no_skill,
        model_kwargs=_model_kwargs,
    )

    stream = agent.run_streamed(query)

    final_text_chunks: list[str] = []
    activated_skills: dict[str, str] = {}  # skill_name → selection reason

    raw_log_f = None
    if args.raw_log_file:
        raw_log_path = Path(args.raw_log_file).expanduser()
        if not raw_log_path.is_absolute():
            raw_log_path = case_dir / raw_log_path
        raw_log_path = raw_log_path.resolve()
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        raw_log_f = raw_log_path.open("w", encoding="utf-8")

    traj_f = None
    traj_json_path: Path | None = None
    traj_json_steps: list[dict] = []
    if args.trajectory_file:
        traj_path = Path(args.trajectory_file).expanduser()
        if not traj_path.is_absolute():
            traj_path = case_dir / traj_path
        traj_path = traj_path.resolve()
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_f = traj_path.open("w", encoding="utf-8")
        traj_f.write("# Agent Execution Trajectory\n\n> 💡 Note: Auto-generated from stream.\n\n")
        traj_json_path = traj_path.with_suffix(".json")

    traj_text_buf: list[str] = []
    traj_current_tool_name = ""
    traj_current_tool_args = ""

    def _flush_traj_text():
        if traj_text_buf and traj_f:
            text = "".join(traj_text_buf).strip()
            if text:
                traj_f.write(f"### 🤖 Agent Thought:\n{text}\n\n")
                traj_f.flush()
                if traj_json_path is not None:
                    traj_json_steps.append({
                        "step": len(traj_json_steps),
                        "type": "thought",
                        "content": text,
                    })
            traj_text_buf.clear()

    def _flush_traj_tool_call(observation: str | None = None):
        nonlocal traj_current_tool_name, traj_current_tool_args
        if traj_current_tool_name and traj_f:
            name = traj_current_tool_name
            args_str = traj_current_tool_args.strip()
            traj_f.write(f"### 🛠 Tool Call: `{name}`\n```json\n{args_str}\n```\n\n")
            traj_f.flush()
            if traj_json_path is not None:
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except Exception:
                    parsed_args = args_str
                traj_json_steps.append({
                    "step": len(traj_json_steps),
                    "type": "tool_call",
                    "tool": name,
                    "args": parsed_args,
                    "observation": observation,
                })
        traj_current_tool_name = ""
        traj_current_tool_args = ""

    dump_dir = None
    dump_idx = 1
    if args.dump_shell_scripts:
        dump_dir = Path(args.dump_shell_scripts).expanduser()
        if not dump_dir.is_absolute():
            dump_dir = case_dir / dump_dir
        dump_dir = dump_dir.resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)

    max_input_tokens_per_turn = getattr(args, "max_input_tokens_per_turn", 0)
    _last_abort_tool: str = ""   # set by early-abort so post-turn check can report it

    try:
        async for event in stream.stream_events():
            if raw_log_f is not None:
                raw_log_f.write(_format_event(event) + "\n")
                raw_log_f.flush()

            # Per-turn input-token budget check — fires on each completed LLM response
            if max_input_tokens_per_turn > 0:
                try:
                    if getattr(event, "type", None) == "raw_response_event":
                        data = getattr(event, "data", None)
                        if getattr(data, "type", None) == "response.completed":
                            turn_input = getattr(getattr(data, "response", None), "usage", None)
                            turn_input = getattr(turn_input, "input_tokens", 0) if turn_input else 0
                            if turn_input > max_input_tokens_per_turn:
                                tool_hint = f" (last large tool: '{_last_abort_tool}')" if _last_abort_tool else ""
                                abort_msg = (
                                    f"[ABORT] Turn input tokens ({turn_input:,}) exceeded "
                                    f"--max-input-tokens-per-turn ({max_input_tokens_per_turn:,})"
                                    f"{tool_hint}. The LLM received a large tool output and the "
                                    f"run was terminated after that turn completed."
                                )
                                print(f"\n[!] {abort_msg}", flush=True)
                                if traj_f:
                                    traj_f.write(f"\n> **[PIPELINE ABORT — post-turn]** {abort_msg}\n\n")
                                    traj_f.flush()
                                break
                except Exception:
                    pass

            if traj_f is not None:
                try:
                    run_item_name = getattr(event, "name", None) or (isinstance(event, dict) and event.get("name")) or ""
                except Exception:
                    run_item_name = ""
                e_type = _get_event_type(event)

                if run_item_name == "tool_output" or "tool_output" in e_type:
                    _flush_traj_text()

                    # Extract observation before flushing the tool call so it
                    # can be stored alongside the tool call in the JSON format.
                    obs = ""
                    if isinstance(event, dict):
                        obs = event.get("output") or event.get("content") or ""
                        if not obs and "item" in event and isinstance(event["item"], dict):
                            raw = event["item"].get("raw_item")
                            if isinstance(raw, dict):
                                obs = raw.get("output") or raw.get("content") or ""
                                if not obs:
                                    outdata = raw.get("output_data") or raw.get("data")
                                    if outdata:
                                        obs = str(outdata)
                    else:
                        try:
                            obs = getattr(event, "output", None) or getattr(event, "content", None) or ""
                            if not obs:
                                item = getattr(event, "item", None)
                                if item:
                                    # Handle ToolCallOutputItem
                                    obs = getattr(item, "output", None) or getattr(item, "content", None) or ""
                                    if not obs:
                                        raw = getattr(item, "raw_item", None)
                                        if isinstance(raw, dict):
                                            obs = raw.get("output") or raw.get("content") or ""
                                            if not obs:
                                                outdata = raw.get("output_data") or raw.get("data")
                                                if outdata:
                                                    obs = str(outdata)
                                        elif raw:
                                            obs = getattr(raw, "output", None) or getattr(raw, "content", None) or ""
                        except Exception:
                            pass

                    obs_str = str(obs).strip() if obs else "(empty or unextractable tool output)"

                    # EARLY ABORT: check BEFORE next LLM call, using tool output size as proxy
                    if max_input_tokens_per_turn > 0:
                        estimated_tokens = len(obs_str) // 4  # rough: 1 token ≈ 4 chars
                        if estimated_tokens > max_input_tokens_per_turn:
                            tool_name = traj_current_tool_name or "(unknown)"
                            tool_args = traj_current_tool_args or ""
                            abort_msg = (
                                f"[ABORT] Tool output too large to pass to LLM: "
                                f"tool='{tool_name}', args={tool_args!r}, "
                                f"~{estimated_tokens:,} estimated tokens ({len(obs_str):,} chars) "
                                f"exceeds --max-input-tokens-per-turn ({max_input_tokens_per_turn:,}). "
                                f"The agent run was terminated; the LLM never saw this output."
                            )
                            print(f"\n[!] {abort_msg}", flush=True)
                            _last_abort_tool = tool_name
                            # Write abort into trajectory so gap agent can read the cause
                            obs_preview = obs_str[:400] + f"\n... ({len(obs_str):,} chars total)" if len(obs_str) > 400 else obs_str
                            abort_obs = f"{abort_msg}\n\nFirst 400 chars of output:\n{obs_preview}"
                            if traj_current_tool_name:
                                _flush_traj_tool_call(observation=abort_obs)
                            if traj_f:
                                traj_f.write(f"\n> **[PIPELINE ABORT — early]** {abort_msg}\n\n")
                                traj_f.flush()
                            break

                    max_len = 2000
                    if len(obs_str) > max_len:
                        obs_str = obs_str[:max_len] + f"\n... (truncated {len(obs_str) - max_len} chars)"

                    # Flush associated tool call (if pending) and write obs
                    if traj_current_tool_name:
                        _flush_traj_tool_call(observation=obs_str)
                    elif traj_json_path is not None:
                        # If flushed already (e.g. parallel tools or separated by thought),
                        # attach to the earliest flushed tool call missing an observation.
                        for step in traj_json_steps:
                            if step.get("type") == "tool_call" and step.get("observation") is None:
                                step["observation"] = obs_str
                                break
                    
                    if traj_f:
                        traj_f.write(f"### 👁 Observation:\n```text\n{obs_str}\n```\n---\n\n")
                        traj_f.flush()

                elif _looks_like_tool_event(event):
                    _flush_traj_text()
                    t_name, t_args = _extract_tool_call_name_and_args(event)
                    if t_name:
                        if traj_current_tool_name and traj_current_tool_name != t_name:
                            _flush_traj_tool_call()
                        traj_current_tool_name = t_name
                    if t_args:
                        if len(t_args) >= len(traj_current_tool_args):
                            traj_current_tool_args = t_args
                        else:
                            if "{" in traj_current_tool_args and t_args.startswith("{"):
                                traj_current_tool_args = t_args
                            else:
                                traj_current_tool_args += t_args
                else:
                    delta = _extract_text_delta(event)
                    if delta:
                        _flush_traj_tool_call()
                        traj_text_buf.append(delta)

            # Optional: dump python scripts executed via `shell` tool.
            # We support:
            #   1) python - <<'PY' ... PY
            #   2) python -c '...'
            #   3) heredoc written to a .py file (cat > x.py <<'PY' ...)
            # And we attempt to snapshot executed .py files (e.g., /tmp/*.py) if they still exist.
            if dump_dir is not None and _looks_like_tool_event(event):
                tool_name, tool_args = _extract_tool_call_name_and_args(event)
                if tool_name == "shell":
                    cmds = _extract_shell_commands_from_args(tool_args)

                    for cmd in cmds:
                        # (3) heredoc to .py file
                        for target, body in _extract_writefile_heredoc(cmd):
                            out_path = dump_dir / f"shell_{dump_idx:03d}_writefile.py"
                            header = (
                                "# Extracted from SkillAgent `shell` tool call (heredoc -> file)\n"
                                f"# original_target: {target}\n"
                                f"# case_dir: {case_dir}\n"
                                f"# inputs: {', '.join(f.name for f in input_files)}\n"
                                f"# out_dir: {out_dir}\n"
                                f"# name_prefix: {_safe_slug(args.name_prefix)}\n"
                                "\n"
                            )
                            out_path.write_text(header + body + "\n", encoding="utf-8")
                            dump_idx += 1

                        # (1) python - heredoc
                        for body in _extract_python_heredoc_bodies(cmd):
                            out_path = dump_dir / f"shell_{dump_idx:03d}_heredoc.py"
                            header = (
                                "# Extracted from SkillAgent `shell` tool call (python heredoc)\n"
                                f"# case_dir: {case_dir}\n"
                                f"# inputs: {', '.join(f.name for f in input_files)}\n"
                                f"# out_dir: {out_dir}\n"
                                f"# name_prefix: {_safe_slug(args.name_prefix)}\n"
                                "\n"
                            )
                            out_path.write_text(header + body + "\n", encoding="utf-8")
                            dump_idx += 1

                        # (2) python -c
                        for body in _extract_python_c_snippets(cmd):
                            out_path = dump_dir / f"shell_{dump_idx:03d}_python_c.py"
                            header = (
                                "# Extracted from SkillAgent `shell` tool call (python -c)\n"
                                f"# case_dir: {case_dir}\n"
                                f"# inputs: {', '.join(f.name for f in input_files)}\n"
                                f"# out_dir: {out_dir}\n"
                                f"# name_prefix: {_safe_slug(args.name_prefix)}\n"
                                "\n"
                            )
                            out_path.write_text(header + body + "\n", encoding="utf-8")
                            dump_idx += 1

                    # Attempt to snapshot executed .py files if present (best-effort).
                    # Only do this on tool output events to maximize chance file exists.
                    try:
                        run_item_name = getattr(event, "name", None)
                    except Exception:
                        run_item_name = None
                    if run_item_name == "tool_output" or (
                        isinstance(event, dict) and event.get("name") == "tool_output"
                    ):
                        for cmd in cmds:
                            for p in _extract_python_exec_paths(cmd):
                                # Resolve relative paths under project root (shell runs in project root).
                                candidate = Path(p)
                                if not candidate.is_absolute():
                                    candidate = (Path(args.project_root) / candidate).resolve()
                                text = _read_text_if_exists(candidate)
                                if not text:
                                    continue
                                out_path = dump_dir / f"shell_{dump_idx:03d}_snapshot.py"
                                header = (
                                    "# Snapshot of executed .py file observed in `shell` command\n"
                                    f"# original_path: {p}\n"
                                    f"# resolved_path: {candidate}\n"
                                    f"# case_dir: {case_dir}\n"
                                    "\n"
                                )
                                out_path.write_text(header + text + "\n", encoding="utf-8")
                                dump_idx += 1

            # Always track activate_skill calls regardless of display mode
            if _looks_like_tool_event(event):
                _t_name, _t_args = _extract_tool_call_name_and_args(event)
                if _t_name == "activate_skill":
                    _sn = _extract_skill_name_from_args(_t_args)
                    if _sn:
                        _reason = _extract_skill_reason_from_args(_t_args)
                        activated_skills[_sn] = _reason

            if args.mode == "raw":
                print(_format_event(event), flush=True)
                continue

            if args.mode == "tools":
                if _looks_like_tool_event(event):
                    print(_format_event(event), flush=True)
                continue

            if args.mode == "skills":
                if _looks_like_tool_event(event):
                    # Only show activate_skill tool calls (skill usage)
                    tool_name, tool_args = _extract_tool_call_name_and_args(event)
                    if tool_name == "activate_skill":
                        skill_name = _extract_skill_name_from_args(tool_args)
                        if skill_name:
                            print(f"[activate_skill] {skill_name}", flush=True)
                        else:
                            # Fallback: print raw args if parsing fails
                            print(f"[activate_skill] args={tool_args}", flush=True)
                continue

            if args.mode == "messages":
                # We cannot access hidden chain-of-thought, but we can show the
                # agent-visible textual narration if the model outputs it.
                if _looks_like_tool_event(event):
                    continue
                delta = _extract_text_delta(event)
                if delta:
                    print(delta, end="", flush=True)
                continue

            # args.mode == "final"
            delta = _extract_text_delta(event)
            if delta:
                final_text_chunks.append(delta)
    except Exception as _stream_exc:
        # MaxTurnsExceeded, UserError, and other SDK exceptions are terminal but
        # normal conditions (not bugs). Print a warning and continue to cost/cleanup.
        _exc_name = type(_stream_exc).__name__
        print(f"\n[!] Stream ended with {_exc_name}: {_stream_exc}", flush=True)
    finally:
        if raw_log_f is not None:
            raw_log_f.close()
        if traj_f is not None:
            _flush_traj_text()
            _flush_traj_tool_call()
            traj_f.close()
        if traj_json_path is not None:
            traj_json_path.write_text(
                json.dumps(
                    {"format": "trajectory_json_v1", "steps": traj_json_steps},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        # Write skills_used ALWAYS — even if the stream raised an exception
        # (e.g. MaxTurnsExceeded) so downstream detect-skills still works.
        if args.skills_used_file:
            try:
                skills_out = Path(args.skills_used_file).expanduser()
                if not skills_out.is_absolute():
                    skills_out = case_dir / skills_out
                skills_out = skills_out.resolve()
                skills_out.parent.mkdir(parents=True, exist_ok=True)
                import json as _json_mod
                skills_out.write_text(_json_mod.dumps(
                    {k: activated_skills[k] for k in sorted(activated_skills)}, indent=2
                ), encoding="utf-8")
            except Exception:
                pass

    if args.mode == "skills":
        if activated_skills:
            skills = ", ".join(sorted(activated_skills.keys()))
            print(f"Activated skills: {skills}", flush=True)
        else:
            print("(No activate_skill events observed. Try --mode tools or --mode raw.)", flush=True)

    if args.mode == "final":
        final_text = "".join(final_text_chunks).strip()
        if final_text:
            print(final_text, flush=True)
        else:
            print(
                "(No text output captured from stream events. Try --mode raw to inspect event shapes.)",
                flush=True,
            )

    # ── Cost tracking ─────────────────────────────────────────────────────────
    from cost_tracker import CostTracker
    tracker = CostTracker(model=args.model, label=args.cost_label)
    tracker.observe(stream)
    tracker.print_summary()
    if args.cost_file:
        cost_path = Path(args.cost_file).expanduser()
        if not cost_path.is_absolute():
            cost_path = case_dir / cost_path
        tracker.save(cost_path.resolve())


if __name__ == "__main__":
    asyncio.run(main())
