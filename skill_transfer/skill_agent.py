"""
Agent Skills implementation mirroring Gemini CLI's architecture.

Uses the OpenAI Agents SDK as the underlying agent framework while following
the exact same prompts, XML formats, tool schemas, and working logic as
Gemini CLI's open-source implementation.

References:
  - Gemini CLI source: https://github.com/google-gemini/gemini-cli
  - Agent Skills spec: https://agentskills.io
  - OpenAI Agents SDK:  https://openai.github.io/openai-agents-python/

Architecture (mirrors Gemini CLI):
  1. Discovery  — scan skill directories for SKILL.md, parse YAML frontmatter
  2. Metadata   — inject name+description as XML into system prompt (~100 tokens/skill)
  3. Activation — model calls activate_skill tool → full instructions returned as XML
  4. Execution  — model follows instructions using standard tools (shell, read, write, etc.)

Usage::

    from skill_agent import SkillAgent

    agent = SkillAgent(skills_dir=".claude/skills", model="gpt-5.4")
    result = agent.run_streamed("Read my PDF and summarise it")
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from agents import Agent, Runner, FunctionTool, function_tool
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.apply_diff import apply_diff
from agents.result import RunResult, RunResultStreaming
from agents.tool_context import ToolContext


# ═══════════════════════════════════════════════════════════════════════════
# Data model  (mirrors Gemini CLI's SkillDefinition — skillLoader.ts:17-30)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SkillDefinition:
    """A discovered skill: metadata + body + location on disk."""

    name: str           # From YAML frontmatter
    description: str    # From YAML frontmatter
    location: str       # Absolute path to the SKILL.md file
    body: str           # Markdown content after the frontmatter
    disabled: bool = False
    is_builtin: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# SKILL.md parser  (mirrors Gemini CLI's skillLoader.ts)
# ═══════════════════════════════════════════════════════════════════════════

# Exact regex from Gemini CLI — skillLoader.ts:32-33
FRONTMATTER_REGEX = re.compile(
    r"^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n([\s\S]*))?"
)


def _parse_frontmatter_yaml(
    content: str,
) -> Optional[dict[str, str]]:
    """Parse YAML frontmatter (primary method).

    Mirrors Gemini CLI's parseFrontmatter() — skillLoader.ts:39-49.
    """
    try:
        parsed = yaml.safe_load(content)
        if parsed and isinstance(parsed, dict):
            name = parsed.get("name")
            description = parsed.get("description")
            if isinstance(name, str) and isinstance(description, str):
                return {"name": name, "description": description}
    except yaml.YAMLError:
        pass
    return None


def _parse_frontmatter_simple(
    content: str,
) -> Optional[dict[str, str]]:
    """Fallback line-by-line parser for edge cases (e.g. colons in description).

    Mirrors Gemini CLI's parseSimpleFrontmatter() — skillLoader.ts:65-108.
    """
    lines = content.split("\n")
    name: Optional[str] = None
    description: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i]

        name_match = re.match(r"^\s*name:\s*(.*)$", line)
        if name_match:
            name = name_match.group(1).strip()
            i += 1
            continue

        desc_match = re.match(r"^\s*description:\s*(.*)$", line)
        if desc_match:
            desc_lines = [desc_match.group(1).strip()]
            # Check for multi-line description (indented continuation lines)
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r"^[ \t]+\S", next_line):
                    desc_lines.append(next_line.strip())
                    i += 1
                else:
                    break
            description = " ".join(filter(None, desc_lines))
            i += 1
            continue

        i += 1

    if name is not None and description is not None:
        return {"name": name, "description": description}
    return None


def _parse_frontmatter(content: str) -> Optional[dict[str, str]]:
    """Parse YAML frontmatter with simple fallback.

    Mirrors Gemini CLI's parseFrontmatter() — skillLoader.ts:39-59.
    """
    result = _parse_frontmatter_yaml(content)
    if result:
        return result
    return _parse_frontmatter_simple(content)


def load_skill_from_file(file_path: str) -> Optional[SkillDefinition]:
    """Load a single SKILL.md file into a SkillDefinition.

    Mirrors Gemini CLI's loadSkillFromFile() — skillLoader.ts:162-190.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        match = FRONTMATTER_REGEX.match(content)
        if not match:
            return None

        frontmatter = _parse_frontmatter(match.group(1))
        if not frontmatter:
            return None

        # Sanitize name — same character set as Gemini CLI (skillLoader.ts:183)
        sanitized_name = re.sub(r'[:\\\/<>*?"|]', "-", frontmatter["name"])

        return SkillDefinition(
            name=sanitized_name,
            description=frontmatter["description"],
            location=str(Path(file_path).resolve()),
            body=(match.group(2) or "").strip(),
        )
    except Exception:
        return None


def load_skills_from_dir(dir_path: str) -> list[SkillDefinition]:
    """Discover skills in a directory.

    Matches Gemini CLI's glob patterns: ``['SKILL.md', '*/SKILL.md']``
    — skillLoader.ts:133-134.
    """
    skills: list[SkillDefinition] = []
    abs_path = Path(dir_path).resolve()

    if not abs_path.is_dir():
        return []

    # Pattern 1: SKILL.md directly in the directory
    direct = abs_path / "SKILL.md"
    if direct.is_file():
        skill = load_skill_from_file(str(direct))
        if skill:
            skills.append(skill)

    # Pattern 2: */SKILL.md — one level deep
    for child in sorted(abs_path.iterdir()):
        if child.is_dir():
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                skill = load_skill_from_file(str(skill_md))
                if skill:
                    skills.append(skill)

    return skills


# ═══════════════════════════════════════════════════════════════════════════
# Skill Manager  (mirrors Gemini CLI's skillManager.ts)
# ═══════════════════════════════════════════════════════════════════════════

class SkillManager:
    """Discovers, manages, and serves Agent Skills with progressive disclosure.

    Mirrors Gemini CLI's SkillManager class — skillManager.ts.
    """

    def __init__(self):
        self._skills: list[SkillDefinition] = []
        self._active_skill_names: set[str] = set()

    # -- Discovery -----------------------------------------------------------

    def clear_skills(self) -> None:
        self._skills = []

    def discover_skills(self, *skill_dirs: str | Path) -> None:
        """Discover skills from multiple directories with precedence.

        Later directories override earlier ones when names collide.
        Mirrors Gemini CLI's discoverSkills() — skillManager.ts:47-92.
        """
        self.clear_skills()
        for dir_path in skill_dirs:
            new_skills = load_skills_from_dir(str(dir_path))
            self._add_skills_with_precedence(new_skills)

    def _add_skills_with_precedence(
        self, new_skills: list[SkillDefinition]
    ) -> None:
        """Add skills; later additions override earlier by name.

        Mirrors Gemini CLI's addSkillsWithPrecedence() — skillManager.ts:117-133.
        """
        skill_map = {s.name: s for s in self._skills}
        for skill in new_skills:
            skill_map[skill.name] = skill
        self._skills = list(skill_map.values())

    # -- Accessors -----------------------------------------------------------

    def get_skills(self) -> list[SkillDefinition]:
        """Return active (non-disabled) skills."""
        return [s for s in self._skills if not s.disabled]

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Case-insensitive skill lookup.

        Mirrors Gemini CLI — skillManager.ts:159-164.
        """
        name_lower = name.lower()
        for s in self._skills:
            if s.name.lower() == name_lower:
                return s
        return None

    def get_skill_names(self) -> list[str]:
        """Return names of all active skills."""
        return [s.name for s in self.get_skills()]

    # -- Activation ----------------------------------------------------------

    def activate_skill(self, name: str) -> None:
        """Mark a skill as activated (state tracking).

        Mirrors Gemini CLI — skillManager.ts:166-168.
        """
        self._active_skill_names.add(name)

    def is_skill_active(self, name: str) -> bool:
        return name in self._active_skill_names

    # -- Skill body access ---------------------------------------------------

    def get_skill_body(self, name: str) -> str | None:
        """Get the full body of a skill."""
        skill = self.get_skill(name)
        if not skill:
            return None
        return skill.body

    # -- Backward compatibility (for try_workflow.py) ------------------------

    @property
    def catalog(self) -> list[dict[str, str]]:
        """Lightweight list of {name, description} for every discovered skill."""
        return [
            {"name": s.name, "description": s.description}
            for s in self.get_skills()
        ]

    def catalog_prompt_fragment(self) -> str:
        """Format the skill catalog + mandate as a prompt fragment."""
        skills = self.get_skills()
        parts = []
        skills_xml = render_agent_skills(skills)
        if skills_xml:
            parts.append(skills_xml)
        mandate = mandate_skill_guidance(len(skills) > 0)
        if mandate:
            parts.append(mandate)
        return "\n\n".join(parts)

    def list_names(self) -> list[str]:
        return self.get_skill_names()

    def refresh(self) -> None:
        """Re-discover skills (convenience for existing callers)."""
        # Caller must call discover_skills() again with the directories
        pass


# ═══════════════════════════════════════════════════════════════════════════
# System prompt generation  (mirrors Gemini CLI's snippets.ts)
# ═══════════════════════════════════════════════════════════════════════════

def render_agent_skills(skills: list[SkillDefinition]) -> str:
    """Render available skills as XML for the system prompt.

    Verbatim from Gemini CLI's renderAgentSkills() — snippets.ts:242-262.
    """
    if not skills:
        return ""

    skills_xml = "\n".join(
        f"  <skill>\n"
        f"    <name>{s.name}</name>\n"
        f"    <description>{s.description}</description>\n"
        f"    <location>{s.location}</location>\n"
        f"  </skill>"
        for s in skills
    )

    return (
        "# Available Agent Skills\n"
        "\n"
        "You have access to the following specialized skills. "
        "To activate a skill and receive its detailed instructions, "
        "call the `activate_skill` tool with the skill's name.\n"
        "\n"
        "<available_skills>\n"
        f"{skills_xml}\n"
        "</available_skills>"
    )


def mandate_skill_guidance(
    has_skills: bool,
    require_skill: bool = False,
    multi_skill: bool = False,
) -> str:
    """Render the skill guidance mandate for the system prompt."""
    if not has_skills:
        return ""

    rules = (
        "1. **Always activate before acting.** When a task clearly matches a "
        "skill's description, call `activate_skill` *before* writing any code or "
        "executing any commands.  Never attempt the task from general knowledge "
        "when a skill exists for it.\n"
        "\n"
        "2. **Follow activated instructions strictly.** Once a skill is activated, "
        "treat both `<instructions>` and `<supplementary_files>` as authoritative "
        "expert guidance.  Prioritize these over your own defaults for the "
        "duration of the task.\n"
        "\n"
        "3. **Use listed scripts.** When the instructions direct you to run a "
        "script, use the exact absolute path from `<runnable_scripts>` — never "
        "guess or construct the path yourself.\n"
        "\n"
        "4. **Skills complement, not replace, core standards.** Follow skill "
        "guidance strictly while continuing to uphold safety and security "
        "principles."
    )

    extra_rules: list[str] = []
    if require_skill:
        extra_rules.append(
            "5. **MANDATORY: You MUST call `activate_skill` at least once** before "
            "writing any code, running any shell commands, or producing any output. "
            "If no skill is a perfect match, activate the most relevant available "
            "skill.  Skipping skill activation entirely is not allowed."
        )
    if multi_skill:
        extra_rules.append(
            "6. **You may activate multiple skills.** The task may span "
            "more than one domain — for example, document search AND statistical "
            "analysis.  Call `activate_skill` for each relevant skill before "
            "attempting the work it covers and needs.  There is no one-skill limit."
        )

    full_rules = rules + ("\n\n" + "\n\n".join(extra_rules) if extra_rules else "")

    return (
        "# How Agent Skills Work\n"
        "\n"
        "Agent Skills are modular, reusable capability packages that give you "
        "domain-specific expertise — workflows, best practices, and executable "
        "scripts — that go beyond your general training.\n"
        "\n"
        "## Three-level progressive loading\n"
        "\n"
        "Skills load on demand to keep context usage minimal:\n"
        "\n"
        "- **Level 1 — Metadata (always present):** Each skill's `name` and "
        "`description` are already in your system prompt (the `<available_skills>` "
        "section above).  No action needed — you already know what skills exist.\n"
        "\n"
        "- **Level 2 — Instructions (loaded when you call `activate_skill`):** "
        "Calling `activate_skill` reads the skill's `SKILL.md` and any "
        "supplementary documentation files (e.g. `reference.md`, `forms.md`) "
        "and returns them all inside `<activated_skill>` tags.  The `<instructions>` "
        "block contains the core procedural workflow; `<supplementary_files>` "
        "contains supporting reference material.  Treat both as authoritative.\n"
        "\n"
        "- **Level 3 — Executable scripts (run on demand):** Executable scripts "
        "are listed with their **absolute paths** in `<runnable_scripts>`. "
        "Always use those exact paths — never construct or guess script paths "
        "yourself.  Run them via `shell` only when the instructions direct you to.\n"
        "\n"
        "## Rules for using skills\n"
        "\n"
        + full_rules
    )


# ═══════════════════════════════════════════════════════════════════════════
# Folder structure helper  (mirrors Gemini CLI's getFolderStructure.ts)
# ═══════════════════════════════════════════════════════════════════════════

_IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def get_folder_structure(skill_dir: str, max_items: int = 200) -> str:
    """BFS listing of a skill's directory, capped at *max_items*.

    Mirrors Gemini CLI's getFolderStructure() — getFolderStructure.ts.
    """
    root = Path(skill_dir)
    if not root.is_dir():
        return "(empty)"

    entries: list[str] = []
    for p in sorted(root.rglob("*")):
        if len(entries) >= max_items:
            entries.append("... (truncated)")
            break
        # Skip ignored directories and their contents
        if any(part in _IGNORED_DIRS for part in p.parts):
            continue
        # Skip SKILL.md backup files (e.g. SKILL.md.bak.20260322_153000)
        if p.is_file() and ".bak." in p.name:
            continue
        rel = p.relative_to(root)
        if p.is_dir():
            entries.append(f"{rel}/")
        else:
            entries.append(str(rel))

    return "\n".join(entries) if entries else "(empty)"


# ═══════════════════════════════════════════════════════════════════════════
# activate_skill tool
# (mirrors Gemini CLI's activate-skill.ts + dynamic-declaration-helpers.ts)
# ═══════════════════════════════════════════════════════════════════════════

def build_activate_skill_tool(mgr: SkillManager) -> FunctionTool:
    """Build the activate_skill function tool with enum-constrained schema.

    Uses ``FunctionTool`` directly (not ``@function_tool``) to control the
    JSON schema — specifically the ``enum`` constraint on skill names.

    Description: verbatim from Gemini CLI — dynamic-declaration-helpers.ts:138-165.
    Response XML: verbatim from Gemini CLI — activate-skill.ts:108-151.
    """
    skill_names = mgr.get_skill_names()

    # -- Description (matches Gemini CLI exactly) ----------------------------
    available_hint = (
        f" (Available: {', '.join(repr(n) for n in skill_names)})"
        if skill_names
        else ""
    )
    description = (
        f"Activates a specialized agent skill by name{available_hint}. "
        "Returns the skill's instructions wrapped in `<activated_skill>` "
        "tags. These provide specialized guidance for the current task. "
        "Use this when you identify a task that matches a skill's "
        "description. ONLY use names exactly as they appear in the "
        "`<available_skills>` section."
    )

    # -- JSON schema with enum constraint (matches Gemini CLI's z.enum()) ----
    if skill_names:
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": skill_names,
                    "description": "The name of the skill to activate.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One sentence explaining why this skill's description matches "
                        "the current task better than any other available skill."
                    ),
                },
            },
            "required": ["name", "reason"],
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "No skills are currently available.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this skill matches the current task.",
                },
            },
            "required": ["name", "reason"],
            "additionalProperties": False,
        }

    # -- Invocation handler --------------------------------------------------
    async def on_invoke(_ctx: ToolContext[Any], args_json: str) -> str:
        args = json.loads(args_json)
        skill_name = args["name"]
        skill = mgr.get_skill(skill_name)

        # Error case — matches Gemini CLI's error format (activate-skill.ts:117-124)
        if skill is None:
            available = ", ".join(mgr.get_skill_names())
            return (
                f'Error: Skill "{skill_name}" not found. '
                f"Available skills are: {available}"
            )

        mgr.activate_skill(skill_name)

        skill_dir = Path(skill.location).parent
        folder_structure = get_folder_structure(str(skill_dir))

        # Build absolute-path listing for every .py script in the skill dir
        py_scripts = sorted(skill_dir.rglob("*.py"))
        if py_scripts:
            script_lines = "\n".join(f"    {p}" for p in py_scripts)
            runnable_scripts_xml = (
                "\n\n  <runnable_scripts>\n"
                "    <!-- Use these EXACT absolute paths when invoking scripts -->\n"
                f"{script_lines}\n"
                "  </runnable_scripts>"
            )
        else:
            runnable_scripts_xml = ""

        # Auto-load supplementary documentation files (non-SKILL.md .md files)
        supplementary_parts: list[str] = []
        for md_file in sorted(skill_dir.glob("*.md")):
            if md_file.name.upper() == "SKILL.MD":
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                supplementary_parts.append(
                    f'  <file name="{md_file.name}">\n'
                    f"{content}\n"
                    f"  </file>"
                )
            except Exception:
                pass

        supplementary_xml = ""
        if supplementary_parts:
            supplementary_xml = (
                "\n\n  <supplementary_files>\n"
                "    <!-- These files are part of the skill. Follow any instructions\n"
                "         or patterns they define alongside the main instructions. -->\n"
                + "\n".join(supplementary_parts)
                + "\n  </supplementary_files>"
            )

        return (
            f'<activated_skill name="{skill_name}">\n'
            f"  <instructions>\n"
            f"    {skill.body}\n"
            f"  </instructions>\n"
            f"\n"
            f"  <available_resources>\n"
            f"    {folder_structure}\n"
            f"  </available_resources>"
            f"{runnable_scripts_xml}"
            f"{supplementary_xml}\n"
            f"</activated_skill>"
        )

    return FunctionTool(
        name="activate_skill",
        description=description,
        params_json_schema=schema,
        on_invoke_tool=on_invoke,
        strict_json_schema=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ApplyPatchTool editor  (from oai_skill_agent.py — unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class LocalFileEditor:
    """ApplyPatchEditor implementation that applies V4A diffs to local files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def _validate_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.resolve()
        if not str(p).startswith(str(self.project_root)):
            raise ValueError(
                f"Path '{path_str}' escapes the project directory."
            )
        return p

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File already exists: {operation.path}",
            )
        try:
            content = apply_diff("", operation.diff or "", mode="create")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ApplyPatchResult(
                status="completed",
                output=f"Created {operation.path} ({len(content)} chars)",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File not found: {operation.path}",
            )
        try:
            original = path.read_text(encoding="utf-8")
            updated = apply_diff(original, operation.diff or "")
            path.write_text(updated, encoding="utf-8")
            return ApplyPatchResult(
                status="completed",
                output=f"Updated {operation.path}",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File not found: {operation.path}",
            )
        try:
            path.unlink()
            return ApplyPatchResult(
                status="completed",
                output=f"Deleted {operation.path}",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# SkillAgent — the main public API
# ═══════════════════════════════════════════════════════════════════════════

class SkillAgent:
    """An OpenAI SDK Agent with Agent Skills support mirroring Gemini CLI.

    The implementation follows Gemini CLI's exact pattern:
      1. Skills metadata (name+description) injected as XML into system prompt
      2. Model calls ``activate_skill`` tool to load full instructions on demand
      3. Instructions returned as ``<activated_skill>`` XML in tool response
      4. Model follows instructions using standard tools (shell, read, write, etc.)

    Args:
        skills_dir:    Path(s) to skill directories. A single path or a list
                       of paths.  When multiple paths are given, later paths
                       have higher precedence (can override earlier skills
                       with the same name).
        model:         OpenAI model identifier (e.g. ``"gpt-5.4"``, ``"gpt-5-mini"``).
        system_prompt: Optional base system prompt.  The skill catalog and
                       mandate are always appended automatically.
        project_root:  Working directory for shell and file tools.
                       Defaults to the current working directory.
        max_turns:     Maximum agent-loop iterations per run.
        model_kwargs:  Extra keyword arguments forwarded to the underlying
                       ``agents.Agent`` constructor.
    """

    def __init__(
        self,
        skills_dir: str | Path | list[str | Path],
        model: str = "gpt-5.4-mini",
        system_prompt: str | None = None,
        project_root: str | Path | None = None,
        shell_cwd: str | Path | None = None,
        max_turns: int = 10,
        model_kwargs: dict | None = None,
        include_skills: list[str] | None = None,
        require_skill: bool = False,
        multi_skill: bool = False,
        no_skill: bool = False,
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.shell_cwd = Path(shell_cwd) if shell_cwd else self.project_root
        self.max_turns = max_turns
        self.model = model
        # No-skill mode (clean control): ignore any discovered skills AND drop the
        # skill *mechanism* — the activate_skill tool is not registered, the system
        # prompt carries no skill catalog/mandate, and read_file does not refer the
        # agent to a (nonexistent) skill for binary/structured files. Isolates the
        # effect of skill CONTENT from the harness's skill scaffolding.
        self.no_skill = no_skill

        # -- Phase 1: Discovery ------------------------------------------------
        self.skill_manager = SkillManager()
        dirs = skills_dir if isinstance(skills_dir, list) else [skills_dir]
        self.skill_manager.discover_skills(*dirs)

        # -- Optional filter: only keep named skills ---------------------------
        if include_skills is not None:
            allowed = {n.lower() for n in include_skills}
            self.skill_manager._skills = [
                s for s in self.skill_manager._skills
                if s.name.lower() in allowed
            ]

        # -- No-skill mode: discard all discovered skills so the catalog and
        #    mandate render empty (works even if skills_dir is populated). --
        if no_skill:
            self.skill_manager._skills = []

        # -- Build tools -------------------------------------------------------
        self.tools = self._build_tools()
        self.tool_names = [
            t.name if hasattr(t, "name") else type(t).__name__
            for t in self.tools
        ]

        # -- Build system prompt (mirrors Gemini CLI's promptProvider.ts) ------
        skills = self.skill_manager.get_skills()

        base = system_prompt or (
            (
                # No-skill base: no mention of skills / activate_skill.
                "You are a capable assistant that completes tasks using tools.\n\n"
                if no_skill else
                "You are a capable assistant that completes tasks using tools and "
                "Agent Skills.  Agent Skills are modular capability packages — each "
                "skill bundles domain-specific instructions, workflows, and optional "
                "executable scripts into a directory on the filesystem.  When a "
                "user's request matches a skill's domain, you MUST activate that "
                "skill first (via `activate_skill`) and then follow its instructions "
                "exactly, rather than relying on your general defaults.\n\n"
            ) +
            "CRITICAL: Writing code or commands in your reasoning does NOT execute "
            "them.  Every shell command and every Python script you intend to run "
            "MUST be issued through the `shell` tool.  Never write code together "
            "with its expected output inside a reasoning step as a substitute for "
            "making the actual `shell` tool call.\n\n"
            "CRITICAL: Always use absolute paths in all tool calls and in any "
            "scripts you write.  Never use bare filenames or relative paths — "
            "they resolve differently depending on which tool you use."
        )

        skills_section = render_agent_skills(skills)
        mandate = mandate_skill_guidance(len(skills) > 0, require_skill=require_skill, multi_skill=multi_skill)

        parts = [base]
        if skills_section:
            parts.append(skills_section)
        if mandate:
            parts.append(mandate)

        self.system_prompt = "\n\n".join(parts)

        # -- Build the underlying OpenAI Agent ---------------------------------
        self.agent = Agent(
            name="SkillAgent",
            instructions=self.system_prompt,
            tools=self.tools,
            model=self.model,
            **(model_kwargs or {}),
        )

    # -- Tool factory -------------------------------------------------------

    def _build_tools(self) -> list:
        """Create all agent tools.

        - ``activate_skill``: built via ``FunctionTool`` for enum schema control
        - Standard tools (read, write, glob, grep): built via ``@function_tool`` closures
        - ``shell`` + ``apply_patch``: SDK built-in tools
        """
        mgr = self.skill_manager
        root = self.project_root
        shell_root = self.shell_cwd
        no_skill = self.no_skill

        # --- activate_skill (Gemini CLI pattern, enum-constrained) ---
        # Built unconditionally but only included in the tool list when skills
        # are enabled (see the return at the end of _build_tools).
        activate_tool = build_activate_skill_tool(mgr)

        # --- Standard tools (closures over `root`) ---

        # File extensions that require skill activation (binary/structured formats)
        _skill_only_extensions = {
            ".pdf", ".xlsx", ".xls", ".xlsm",
            ".pptx", ".ppt", ".docx", ".doc",
        }

        @function_tool
        def read_file(file_path: str) -> str:
            """Read the contents of a TEXT file (e.g., .py, .txt, .json, .csv, .md).

            This tool is for plain-text files ONLY. It CANNOT read binary or
            structured files like PDF, XLSX, PPTX, or DOCX. For those file
            types, you MUST activate the corresponding skill first and use
            the code examples it provides.

            Args:
                file_path: Absolute path to the file. Always use an absolute path.
            """
            p = Path(file_path)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            if not p.exists():
                return f"Error: file '{file_path}' not found."
            if p.suffix.lower() in _skill_only_extensions:
                if no_skill:
                    return (
                        f"Error: '{p.suffix}' files are binary/structured and cannot "
                        f"be read as plain text. Use the shell tool with an appropriate "
                        f"library (e.g. openpyxl for .xlsx, pdfplumber for .pdf, "
                        f"python-docx for .docx) to process this file."
                    )
                return (
                    f"Error: '{p.suffix}' files cannot be read as plain text. "
                    f"Activate the appropriate skill (e.g., pdf, xlsx, pptx) "
                    f"and use its code examples to process this file."
                )
            try:
                return p.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Error reading file: {exc}"

        @function_tool
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file, creating parent directories if needed.

            Use this to create or update files.

            Args:
                file_path: Absolute path to write to. Always use an absolute path.
                content: The full text content to write.
            """
            p = Path(file_path)
            if not p.is_absolute():
                p = shell_root / p
            p = p.resolve()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} characters to {p}"
            except Exception as exc:
                return f"Error writing file: {exc}"

        @function_tool
        def glob_files(pattern: str, path: str = "") -> str:
            """Find files matching a glob pattern within a directory.

            Args:
                pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts").
                path: Absolute path to the directory to search in. Leave empty to search the entire project root.
            """
            search_root = (root / path).resolve() if path else root.resolve()
            try:
                matches = sorted(search_root.glob(pattern))
                results = [
                    str(m.relative_to(root.resolve()))
                    for m in matches
                    if m.is_file()
                ]
                if not results:
                    return "No files matched the pattern."
                return "\n".join(results)
            except Exception as exc:
                return f"Error in glob: {exc}"

        @function_tool
        def grep_files(
            pattern: str, path: str = "", glob_filter: str = ""
        ) -> str:
            """Search file contents for a regex pattern.

            Args:
                pattern: Regular expression pattern to search for.
                path: Absolute path to the directory to search in. Leave empty to search the entire project root.
                glob_filter: Optional glob to filter files (e.g. "*.py").
            """
            search_path = (root / path).resolve() if path else root.resolve()
            cmd = ["grep", "-Ern", pattern, str(search_path)]
            if glob_filter:
                cmd = [
                    "grep", "-Ern", "--include", glob_filter,
                    pattern, str(search_path),
                ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(root),
                )
                output = proc.stdout.strip()
                if not output:
                    return "No matches found."
                if len(output) > 10000:
                    output = output[:10000] + "\n...(truncated)"
                return output
            except subprocess.TimeoutExpired:
                return "Error: grep timed out after 30s."
            except Exception as exc:
                return f"Error in grep: {exc}"

        # --- Shell tool (wrapped as FunctionTool for visibility) ---
        # Uses subprocess.run() directly instead of SDK ShellTool types

        async def shell_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            args = json.loads(args_json)
            commands = args["commands"]
            timeout_ms = args.get("timeout_ms")
            timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120

            outputs = []
            for cmd in commands:
                try:
                    proc = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                        cwd=str(shell_root),
                        env={**os.environ},
                    )
                    cmd_output = f"Command: {cmd}\n"
                    if proc.stdout:
                        cmd_output += f"stdout:\n{proc.stdout}\n"
                    if proc.stderr:
                        cmd_output += f"stderr:\n{proc.stderr}\n"
                    cmd_output += f"exit_code: {proc.returncode}"
                    outputs.append(cmd_output)
                except subprocess.TimeoutExpired:
                    outputs.append(
                        f"Command: {cmd}\nError: timed out after {timeout_sec}s"
                    )
                    break

            return "\n\n".join(outputs) if outputs else "No output"

        shell_tool = FunctionTool(
            name="shell",
            description=(
                "Execute shell commands in the project directory. "
                "Use this to run terminal commands, scripts, tests, build tools, etc. "
                "Each command runs in the project root with the user's environment."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of shell commands to execute sequentially.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds (default: 120000).",
                    },
                },
                "required": ["commands"],
            },
            on_invoke_tool=shell_handler,
            strict_json_schema=False,
        )

        # --- Apply patch tool (wrapped as FunctionTool for visibility) ---
        editor = LocalFileEditor(root)

        async def apply_patch_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            args = json.loads(args_json)
            operations = args["operations"]

            from agents.editor import ApplyPatchOperation

            # Map legacy short names to the SDK's current enum values
            _OP_MAP = {"create": "create_file", "update": "update_file", "delete": "delete_file"}

            results = []
            for op_dict in operations:
                raw_type = op_dict.get("operation_type", "update")
                sdk_type = _OP_MAP.get(raw_type, raw_type)  # pass through if already "xxx_file"

                # ── Update path: prefer reliable old_str/new_str over diff ───
                # When operation_type=update AND old_str is provided, perform
                # an exact-string replacement instead of going through the
                # unified-diff parser. This mirrors the replace_in_file
                # tool — LLMs are far more reliable at
                # producing exact-match strings than well-formed unified diffs.
                if sdk_type == "update_file" and op_dict.get("old_str") is not None:
                    old_str = op_dict["old_str"]
                    new_str = op_dict.get("new_str", "")
                    raw_path = op_dict["path"]
                    p = Path(raw_path)
                    if not p.is_absolute():
                        p = root / raw_path
                    p = p.resolve()
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
                        # Show a few candidate lines so the LLM knows what's nearby
                        snippet = (old_str.splitlines() or [""])[0][:80]
                        results.append(
                            f"[failed] {raw_path}: old_str not found in file. "
                            f"Re-read the file with read_file and copy the exact "
                            f"text to replace. (First line of attempted old_str: "
                            f"{snippet!r})"
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
                    delta = len(updated) - len(original)
                    results.append(
                        f"[success] {raw_path}: replaced 1 occurrence "
                        f"({delta:+d} chars)"
                    )
                    continue

                # ── Legacy diff path (create / delete / update-with-diff) ────
                operation = ApplyPatchOperation(
                    type=sdk_type,
                    path=op_dict["path"],
                    diff=op_dict.get("diff"),
                )

                if sdk_type == "create_file":
                    result = editor.create_file(operation)
                elif sdk_type == "update_file":
                    result = editor.update_file(operation)
                elif sdk_type == "delete_file":
                    result = editor.delete_file(operation)
                else:
                    results.append(
                        f"Error: Unknown operation type '{sdk_type}' for {operation.path}"
                    )
                    continue

                results.append(f"[{result.status}] {result.output}")

            return "\n".join(results) if results else "No operations performed"

        apply_patch_tool = FunctionTool(
            name="apply_patch",
            description=(
                "Apply file operations (create / update / delete).\n"
                "\n"
                "For operation_type='update', PREFER the exact-string replace "
                "mode: provide `old_str` (a verbatim unique snippet from the "
                "current file, copied from a recent read_file output) and "
                "`new_str` (the replacement). This is the most reliable "
                "editing path.\n"
                "\n"
                "The legacy unified-diff mode (provide `diff` instead of "
                "old_str/new_str) is still supported but LLMs frequently "
                "produce malformed diffs that fail to parse — use it only if "
                "you cannot fit the update into a single old_str/new_str pair.\n"
                "\n"
                "For operation_type='create', provide `diff` containing the "
                "full file body. For operation_type='delete', provide only "
                "the path."
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
                                    "description": "Type of operation to perform.",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "File path (absolute or project-relative).",
                                },
                                "old_str": {
                                    "type": "string",
                                    "description": (
                                        "(update only, PREFERRED) Exact "
                                        "string to find — must match exactly "
                                        "one location in the file. Copy "
                                        "verbatim from a recent read_file "
                                        "output, including surrounding "
                                        "context if the string is not unique."
                                    ),
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": (
                                        "(update only, PREFERRED) Replacement "
                                        "string. Pair with old_str. Empty "
                                        "string deletes the matched region."
                                    ),
                                },
                                "diff": {
                                    "type": "string",
                                    "description": (
                                        "Unified diff body. Used for "
                                        "create (full file body), and as "
                                        "fallback for update when old_str/"
                                        "new_str are not provided."
                                    ),
                                },
                            },
                            "required": ["operation_type", "path"],
                        },
                        "description": "List of file operations to perform.",
                    },
                },
                "required": ["operations"],
            },
            on_invoke_tool=apply_patch_handler,
            strict_json_schema=False,
        )

        _tools = [
            read_file,
            write_file,
            glob_files,
            grep_files,
            shell_tool,
            apply_patch_tool,
        ]
        # No-skill mode: omit the activate_skill tool entirely (clean control).
        return _tools if no_skill else [activate_tool, *_tools]

    # -- Rebuild (re-purpose for a different phase) -------------------------

    def rebuild(self, system_prompt: str, name: str = "SkillAgent") -> None:
        """Re-create the underlying Agent with new instructions.

        Shares the same SkillManager and tools — only the Agent object and
        system prompt are replaced.  Use this to switch the agent to a
        different workflow phase (e.g. from task execution to skill evolution).

        Args:
            system_prompt: New base system prompt for the agent.
            name: Display name for the new Agent instance.
        """
        skills = self.skill_manager.get_skills()
        skills_section = render_agent_skills(skills)
        mandate = mandate_skill_guidance(len(skills) > 0)

        parts = [system_prompt]
        if skills_section:
            parts.append(skills_section)
        if mandate:
            parts.append(mandate)

        self.system_prompt = "\n\n".join(parts)
        self.agent = Agent(
            name=name,
            instructions=self.system_prompt,
            tools=self.tools,
            model=self.model,
        )

    # -- Run methods --------------------------------------------------------

    async def run(self, query: str, **kwargs) -> RunResult:
        """Run the agent to completion (async).

        Args:
            query: The user's input query.
            **kwargs: Extra keyword args forwarded to ``Runner.run``.
        """
        return await Runner.run(
            self.agent,
            query,
            max_turns=kwargs.pop("max_turns", self.max_turns),
            **kwargs,
        )

    def run_streamed(self, query: str, **kwargs) -> RunResultStreaming:
        """Start a streamed agent run.

        Returns a ``RunResultStreaming`` whose ``.stream_events()`` async
        iterator yields events in real time.

        Args:
            query: The user's input query.
            **kwargs: Extra keyword args forwarded to ``Runner.run_streamed``.
        """
        return Runner.run_streamed(
            self.agent,
            query,
            max_turns=kwargs.pop("max_turns", self.max_turns),
            **kwargs,
        )
