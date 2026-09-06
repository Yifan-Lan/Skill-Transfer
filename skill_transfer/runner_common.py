"""Shared helpers for the Diagnoser and Patcher runners.

Terminal colours, retry/snapshot utilities, patch-narrative extraction, and the
streaming printer that renders an agent run to stdout and to a trajectory file.
Imported by run_gap_diagnoser.py and run_skill_patcher.py.
"""


from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
import re
import signal
import sys
from pathlib import Path

from openai_setup import setup_openai
setup_openai("gap-agent")

# ── colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[90m"

def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # repository root

_PATCH_TOOLS = {"replace_in_file", "apply_patch"}
# Tool calls included with args only (observation stripped) but not patch tools
_ARGS_ONLY_TOOLS = {"read_prev_case_structures"}


def _format_retry_exception(exc: BaseException, *, max_field_chars: int = 4000) -> str:
    """Return diagnostic details for retryable API/transport exceptions."""
    def _clip(value) -> str:
        text = str(value)
        if len(text) > max_field_chars:
            return text[:max_field_chars] + f"... [truncated {len(text) - max_field_chars} chars]"
        return text

    lines = [
        f"type: {type(exc).__module__}.{type(exc).__name__}",
        f"repr: {_clip(repr(exc))}",
        f"message: {_clip(str(exc))}",
    ]

    for attr in ("status_code", "request_id", "code", "param", "type"):
        if hasattr(exc, attr):
            try:
                value = getattr(exc, attr)
            except Exception as attr_exc:
                value = f"<error reading attr: {attr_exc!r}>"
            if value is not None:
                lines.append(f"{attr}: {_clip(value)}")

    body = getattr(exc, "body", None)
    if body is not None:
        try:
            body_text = json.dumps(body, ensure_ascii=False, default=str)
        except Exception:
            body_text = repr(body)
        lines.append(f"body: {_clip(body_text)}")

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            lines.append(f"response.status_code: {status}")
        headers = getattr(response, "headers", None)
        if headers is not None:
            for key in ("x-request-id", "request-id", "apim-request-id", "x-ms-request-id"):
                try:
                    value = headers.get(key)
                except Exception:
                    value = None
                if value:
                    lines.append(f"response.headers.{key}: {_clip(value)}")
        text = getattr(response, "text", None)
        if text:
            lines.append(f"response.text: {_clip(text)}")

    if exc.__cause__ is not None:
        lines.append(f"cause: {type(exc.__cause__).__module__}.{type(exc.__cause__).__name__}: {_clip(exc.__cause__)}")
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        lines.append(f"context: {type(exc.__context__).__module__}.{type(exc.__context__).__name__}: {_clip(exc.__context__)}")

    return "\n".join(lines)


_STALE_REASONING_ITEM_PATTERNS = (
    "Item with id 'rs_",        # Azure Responses API reasoning-item TTL expiry
    "previous_response_id",     # alternate phrasing for the same root cause
)


def _is_stale_session_error(exc: BaseException) -> bool:
    """Detect the Azure 400 'Item with id rs_... not found' error.

    This happens when the SQLiteSession's previous_response_id chain
    references reasoning items that Azure has garbage-collected after
    their server-side TTL.  Preserving the session DB is counter-
    productive — every subsequent retry will hit the same stale id.
    The caller should delete the session DB and start fresh instead.
    """
    msg = str(exc)
    return any(pat in msg for pat in _STALE_REASONING_ITEM_PATTERNS)


def _archive_retry_artifacts(log_dir: Path, retry_num: int) -> Path:
    """Move the partial attempt's top-level artifacts aside before an internal retry."""
    archive_dir = log_dir / f"api_retry_{retry_num:02d}_failed"
    suffix = 1
    while archive_dir.exists():
        suffix += 1
        archive_dir = log_dir / f"api_retry_{retry_num:02d}_failed_{suffix}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in list(log_dir.iterdir()):
        if path == archive_dir or path.name.startswith("api_retry_"):
            continue
        path.rename(archive_dir / path.name)
    return archive_dir


def _snapshot_files(paths: list[Path]) -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in paths}


def _restore_file_snapshots(snapshots: dict[Path, str]) -> None:
    for path, content in snapshots.items():
        path.write_text(content, encoding="utf-8")


def extract_patch_narrative(traj_json_path: Path) -> Path | None:
    """Extract patch reasoning + edits from a gap agent trajectory JSON.

    Keeps only steps that appear AFTER the gap report is submitted:
      - thought steps (assistant reasoning text)
      - replace_in_file / apply_patch call arguments (no observations)
      - read_prev_case_structures call arguments (no observations) — records
        which cases were checked during STATUS ANALYSIS without the bulky
        prev-structure JSON

    Boundary detection (first match wins):
      1. submit_gap_report tool call (new path)
      2. write_file call whose file_path contains gap_report.json (legacy path)

    Saves to <traj_json_path.parent>/gap_patch_narrative.json.
    Returns the output path, or None if the trajectory is missing/unreadable
    or contains no gap report submission.
    """
    try:
        data = json.loads(traj_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    steps = data.get("steps", [])

    # Find boundary: submit_gap_report (preferred) or legacy write_file
    boundary: int | None = None
    for s in steps:
        if s.get("type") != "tool_call":
            continue
        if s.get("tool") == "submit_gap_report":
            boundary = s["step"]
            break
        if (s.get("tool") == "write_file"
                and "gap_report.json" in str(s.get("args", {}).get("file_path", ""))):
            boundary = s["step"]
            break

    if boundary is None:
        return None

    kept: list[dict] = []
    for s in steps:
        if s["step"] <= boundary:
            continue
        if s.get("type") == "thought":
            kept.append({"type": "thought", "text": s.get("text", "")})
        elif s.get("type") == "tool_call" and s.get("tool") in (_PATCH_TOOLS | _ARGS_ONLY_TOOLS):
            kept.append({
                "type": "tool_call",
                "tool": s["tool"],
                "args": s.get("args", {}),
            })

    out_path = traj_json_path.parent / "gap_patch_narrative.json"
    out_path.write_text(
        json.dumps({"format": "patch_narrative_v1", "steps": kept},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def _v4a_to_unified(diff_text: str, path: str = "") -> str:
    """Convert V4A-style diff to standard unified diff for markdown rendering.

    V4A uses:  |line  for context,  +line / -line  for edits, @@ for hunks,
    and  *** Begin/End Patch  metadata that has no place in a diff block.
    Standard unified diff uses a leading space for context lines.
    """
    out: list[str] = []
    if path:
        out += [f"--- a/{path}", f"+++ b/{path}"]
    for line in diff_text.split("\n"):
        if line in (":", "*** Begin Patch", "*** End Patch"):
            continue
        if line.startswith(("*** Update File:", "*** Create File:", "*** Delete File:")):
            continue
        if line == "@@":                 # bare @@ hunk marker
            out.append("@@ -1 +1 @@")
            continue
        if line.startswith("|"):         # V4A context line → space prefix
            out.append(" " + line[1:])
            continue
        out.append(line)
    return "\n".join(out)


def _fenced(content: str, lang: str = "") -> str:
    """Return a fenced code block whose delimiter can't be broken by content."""
    max_run = max((len(m.group()) for m in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(3, max_run + 1)
    return f"{fence}{lang}\n{content}\n{fence}"

# Re-use the battle-tested event helpers from run_skill_agent
from run_skill_agent import (
    _get_event_type,
    _extract_text_delta,
    _extract_tool_call_name_and_args,
    _extract_tool_call_id,
    _looks_like_tool_event,
)


# ── streaming display ─────────────────────────────────────────────────────────

_OBS_LIMIT       = 1200   # max chars for read_file / write_file observations
_SHELL_OBS_LIMIT = 2000   # max chars for shell observations (longer for script output)
_STRUCT_OBS_LIMIT = 6000  # max chars for read_case_structures display (full data still sent to LLM)
_REPORT_OBS_LIMIT = 4000  # max chars for read_case_validator_report / read_case_history / list_cases
_ARGS_LIMIT      = 800    # max chars for generic tool-call args in the trajectory


def _truncate(text: str, limit: int) -> str:
    """Return text truncated to limit with an omission note if needed."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} chars omitted]"

class StreamPrinter:
    """Stateful pretty-printer for agent stream events."""

    def __init__(self, traj_file=None, raw_log_file=None, traj_json_path=None,
                 preload_steps: bool = False):
        self._thought_buf: list[str] = []
        self._tool_name: str | None = None
        self._tool_call_id: str = ""   # unique call ID; detects parallel calls with the same name
        self._tool_args: list[str] = []
        self._last_read_path: str | None = None   # captured from read_file args
        self._pending_output_tool_name: str | None = None  # fallback for non-JSON display
        self._traj = traj_file
        self._raw  = raw_log_file
        # JSON trajectory (mirrors run_skill_agent.py's trajectory_json_v1 format)
        self._traj_json_path: Path | None = Path(traj_json_path) if traj_json_path else None
        # On resume, preload prior steps so finalize_json accumulates across retry segments.
        self._json_steps: list[dict] = []
        if preload_steps and self._traj_json_path and self._traj_json_path.exists():
            try:
                _prior = json.loads(self._traj_json_path.read_text(encoding="utf-8"))
                self._json_steps = _prior.get("steps", [])
            except Exception:
                pass
        # Queue of (tool_name, json_step_index) for each pending tool call.
        # A deque is required because the LLM may issue parallel tool calls in one
        # response: observations arrive one-by-one and must be matched FIFO to the
        # calls that were streamed earlier.
        self._pending_tool_queue: deque[tuple[str, int | None]] = deque()

    def _write_traj(self, text: str):
        if self._traj:
            self._traj.write(text)
            self._traj.flush()

    def _write_raw(self, event):
        if self._raw:
            self._raw.write(repr(event) + "\n")
            self._raw.flush()

    def flush_thought(self, agent_instance=None):
        text = "".join(self._thought_buf).strip()
        if text:
            print(f"\n{GREEN}[Agent]{RESET}\n{text}")
            self._write_traj(f"\n### 🤖 Agent\n\n{text}\n")
            if agent_instance:
                agent_instance.last_thought = text
            if self._traj_json_path is not None:
                self._json_steps.append({
                    "step": len(self._json_steps),
                    "type": "thought",
                    "text": text,
                })
                self._persist_json()
        self._thought_buf.clear()

    @staticmethod
    def _lang_for_path(path: str) -> str:
        """Return a fenced-code language tag based on file extension."""
        ext = Path(path).suffix.lower()
        return {"json": "json", ".json": "json",
                ".py": "python", ".sh": "bash",
                ".md": "markdown", ".txt": "text",
                ".yaml": "yaml", ".yml": "yaml"}.get(ext, "text")

    def flush_tool(self):
        if not self._tool_name:
            return
        name = self._tool_name
        args_str = "".join(self._tool_args)
        # capture read_file path before args are cleared
        if name == "read_file":
            try:
                self._last_read_path = json.loads(args_str).get("file_path", "")
            except Exception:
                self._last_read_path = None
        else:
            self._last_read_path = None
        print(f"\n{BLUE}[Tool Call]{RESET}: {BOLD}{name}{RESET}")

        if name == "apply_patch":
            try:
                parsed = json.loads(args_str)
                for op in parsed.get("operations", []):
                    print(f"{YELLOW}  target: {op.get('path', '?')}{RESET}")
                    diff = op.get("diff", "")
                    for line in diff.split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            print(f"{GREEN}{line}{RESET}")
                        elif line.startswith("-") and not line.startswith("---"):
                            print(f"{RED}{line}{RESET}")
                        elif line.startswith("@@"):
                            print(f"{CYAN}{line}{RESET}")
                        else:
                            print(line)
            except Exception:
                print(f"{DIM}{args_str}{RESET}")
        elif name == "replace_in_file":
            try:
                parsed = json.loads(args_str)
                print(f"{YELLOW}  target : {parsed.get('path', '?')}{RESET}")
                old_s = parsed.get("old_str", "")
                new_s = parsed.get("new_str", "")
                for line in old_s.split("\n"):
                    print(f"{RED}-{line}{RESET}")
                for line in new_s.split("\n"):
                    print(f"{GREEN}+{line}{RESET}")
            except Exception:
                print(f"{DIM}{args_str}{RESET}")
        elif name == "generate_viz":
            print(f"{DIM}{args_str}{RESET}")
        else:
            print(f"{DIM}{args_str}{RESET}")

        if name == "apply_patch":
            try:
                parsed = json.loads(args_str)
                traj_str = f"\n### 🛠 Tool Call: `{name}`\n{_fenced(json.dumps(parsed, indent=2, ensure_ascii=False), 'json')}\n"
                for op in parsed.get("operations", []):
                    diff_text = op.get("diff", "")
                    path = op.get("path", "unknown")
                    if diff_text:
                        unified = _v4a_to_unified(diff_text, path)
                        traj_str += f"\n> **Target:** `{path}`\n{_fenced(unified, 'diff')}\n"
            except Exception:
                traj_str = f"\n### 🛠 Tool Call: `{name}`\n{_fenced(args_str, 'json')}\n"
            self._write_traj(traj_str)
        elif name == "replace_in_file":
            try:
                parsed = json.loads(args_str)
                path    = parsed.get("path", "unknown")
                old_str = parsed.get("old_str", "")
                new_str = parsed.get("new_str", "")
                diff_body = (
                    f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n"
                    + "\n".join(f"-{l}" for l in old_str.split("\n"))
                    + "\n"
                    + "\n".join(f"+{l}" for l in new_str.split("\n"))
                )
                traj_str = (
                    f"\n### 🛠 Tool Call: `{name}`\n"
                    f"> **Target:** `{path}`\n"
                    f"{_fenced(diff_body, 'diff')}\n"
                )
            except Exception:
                traj_str = f"\n### 🛠 Tool Call: `{name}`\n```json\n{args_str}\n```\n"
            self._write_traj(traj_str)
        else:
            self._write_traj(f"\n### 🛠 Tool Call: `{name}`\n```json\n{args_str}\n```\n")
        self._pending_output_tool_name = name   # fallback when queue is empty
        if self._traj_json_path is not None:
            try:
                parsed_args = json.loads(args_str) if args_str else {}
            except Exception:
                parsed_args = {"raw": args_str}
            step_idx = len(self._json_steps)
            self._json_steps.append({
                "step": step_idx,
                "type": "tool_call",
                "tool": name,
                "args": parsed_args,
                "observation": "",
            })
            self._persist_json()
            self._pending_tool_queue.append((name, step_idx))
        else:
            self._pending_tool_queue.append((name, None))
        self._tool_name = None
        self._tool_call_id = ""
        self._tool_args.clear()

    def handle(self, event, agent_instance=None):
        self._write_raw(event)
        e_type = _get_event_type(event)
        run_item_name = getattr(event, "name", None) or ""

        # Completion
        if "run_complete" in e_type or "response.done" in e_type:
            self.flush_thought(agent_instance)
            self.flush_tool()
            print(f"\n\n{GREEN}{BOLD}[✓] Gap analysis & skill patching complete!{RESET}")
            print("─" * 60)
            final_res = getattr(event, "result", None)
            if final_res:
                txt = getattr(final_res, "final_output", getattr(final_res, "output", ""))
                if txt:
                    print(txt)
                    self._write_traj(f"\n### Final Output\n\n{txt}\n")
            return

        if _looks_like_tool_event(event):
            if run_item_name == "tool_output" or "tool_output" in e_type:
                self.flush_thought(agent_instance)
                self.flush_tool()  # flushes last pending call, pushes it onto the queue
                obs = self._extract_obs(event)

                # Pop FIFO from the queue so parallel tool calls get the right observation.
                # flush_tool() above may have just pushed the final call in a parallel batch.
                if self._pending_tool_queue:
                    t_name, pending_idx = self._pending_tool_queue.popleft()
                else:
                    t_name = self._pending_output_tool_name or ""
                    pending_idx = None

                if t_name in ("read_file", "write_file", "read_case_trajectory"):
                    display_obs = _truncate(obs, _OBS_LIMIT)
                elif t_name == "shell":
                    display_obs = _truncate(obs, _SHELL_OBS_LIMIT)
                elif t_name in ("read_case_structures", "read_prev_case_structures"):
                    display_obs = _truncate(obs, _STRUCT_OBS_LIMIT)
                elif t_name in ("read_case_validator_report", "read_case_history",
                                "list_cases"):
                    display_obs = _truncate(obs, _REPORT_OBS_LIMIT)
                else:
                    display_obs = obs

                tool_label = str(t_name) if t_name else "Tool"
                top_border = f"{BOLD}{YELLOW}╭{'─'*15} 👁  OBSERVATION : {tool_label} {'─'*40}{RESET}"
                bot_border = f"{BOLD}{YELLOW}╰{'─'*75}{RESET}"
                print(f"\n{top_border}\n{display_obs}\n{bot_border}")

                # Fill JSON step observation
                if pending_idx is not None:
                    self._json_steps[pending_idx]["observation"] = display_obs
                    self._persist_json()

                # decorated trajectory format for read_file
                if t_name == "read_file" and self._last_read_path:
                    fname = Path(self._last_read_path).name
                    lang  = self._lang_for_path(self._last_read_path)
                    traj_obs = (
                        f"\n### 👁 Observation (from `read_file`)\n\n"
                        f"> 📄 **`{fname}`** &nbsp;·&nbsp; `{self._last_read_path}`\n\n"
                        f"```{lang}\n{display_obs}\n```\n"
                    )
                else:
                    traj_obs = f"\n### 👁 Observation (from `{tool_label}`)\n```text\n{display_obs}\n```\n"
                self._write_traj(traj_obs)
            else:
                t_name, t_args = _extract_tool_call_name_and_args(event)
                t_call_id = _extract_tool_call_id(event)
                # Flush when the tool name changes OR when the same tool name
                # arrives with a new call ID (parallel calls in one LLM turn).
                is_new_call = bool(t_name) and (
                    t_name != self._tool_name
                    or (t_call_id and t_call_id != self._tool_call_id)
                )
                if is_new_call:
                    self.flush_thought(agent_instance)
                    self.flush_tool()
                    self._tool_name = t_name
                    self._tool_call_id = t_call_id
                    print(f"\n{BLUE}→ Invoking '{t_name}'…{RESET}", flush=True)
                if t_args:
                    joined = "".join(self._tool_args)
                    if len(t_args) >= len(joined):
                        self._tool_args = [t_args]
                    else:
                        self._tool_args.append(t_args)
        else:
            delta = _extract_text_delta(event)
            if delta:
                self._thought_buf.append(delta)
                print(delta, end="", flush=True)

    def _persist_json(self):
        """Atomically write the current step list to trajectory.json.

        Called incrementally after every step mutation so the file on disk
        is always up-to-date — this lets attempt_N/ archives include a
        meaningful .json (previously the writer only flushed on close, so
        a crashed attempt left an empty/missing .json that broke resume
        accumulation).  Atomic via write-temp + os.replace.
        """
        if self._traj_json_path is None or not self._json_steps:
            return
        try:
            self._traj_json_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._traj_json_path.with_suffix(
                self._traj_json_path.suffix + ".tmp"
            )
            tmp_path.write_text(
                json.dumps(
                    {"format": "trajectory_json_v1", "steps": self._json_steps},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._traj_json_path)
        except Exception as exc:
            print(f"[warn] Could not write trajectory JSON: {exc}", file=sys.stderr)

    def finalize_json(self):
        """End-of-stream flush, called by both agent runners."""
        self._persist_json()

    @staticmethod
    def _extract_obs(event) -> str:
        if isinstance(event, dict):
            obs = event.get("output") or event.get("content") or ""
            if not obs and "item" in event and isinstance(event["item"], dict):
                raw = event["item"].get("raw_item")
                if isinstance(raw, dict):
                    obs = raw.get("output") or raw.get("content") or ""
            return str(obs)
        try:
            obs = getattr(event, "output", None) or getattr(event, "content", None) or ""
            if not obs:
                item = getattr(event, "item", None)
                raw  = getattr(item, "raw_item", None) if item else None
                if isinstance(raw, dict):
                    obs = raw.get("output", "") or raw.get("content", "")
                else:
                    obs = getattr(raw, "output", None) or getattr(raw, "content", None) or ""
        except Exception:
            obs = ""
        return str(obs)


# ── main ──────────────────────────────────────────────────────────────────────
