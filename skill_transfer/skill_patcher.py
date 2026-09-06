"""SkillPatcher: the repair step of Skill Transfer.

Reads the ranked gap report produced by GapDiagnoser and applies every patch
hint to the skill in a single consolidated pass.  It does not re-derive which
gaps exist; the diagnosis is taken as given.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.memory.sqlite_session import SQLiteSession

from gap_base import (
    GapAgentBase, CaseContext, LocalFileEditor,
    _skill_backup_path, _expand_skill_listing,
    FAIL_WEIGHT, PASS_WEIGHT, SCORE_THRESHOLD,
)

from skill_agent import (
    SkillManager, load_skills_from_dir, build_activate_skill_tool,
    render_agent_skills, mandate_skill_guidance,
)

logger = logging.getLogger(__name__)

# ─── System-prompt sections ───

_SKILL_ARCHITECTURE_BANNER = """
════════════════════════════════════════════════════════
SKILL ARCHITECTURE — PROGRESSIVE DISCLOSURE
════════════════════════════════════════════════════════

Skills are designed around progressive disclosure: information is loaded in
stages, only when needed, to avoid bloating the agent's context window.

  Level 1 — Metadata (YAML frontmatter, always loaded, ~100 tokens)
    The skill's name and description.  Always in context.  Never patch this
    unless the description is genuinely wrong.

  Level 2 — Instructions (SKILL.md body, loaded when triggered, target ≤5k tokens)
    Core procedural knowledge: required workflows, key technique notes, and the
    most important code example for each operation.  THIS IS WHERE YOU PATCH.
    Keep it lean.  Every sentence in SKILL.md is paid for in context on every
    invocation.  Prefer short, imperative rules over long explanations.

  Level 3 — Support files (documentation files, scripts; loaded on demand)
    Any file in the skill directory beyond SKILL.md — detailed reference docs,
    edge-case examples, executable scripts.  A skill agent reads or runs these
    only when SKILL.md directs it to.  ALL of these files are patchable:
    patch a doc file if its content is the root of failure; patch a script
    file directly if its executable logic is wrong.
"""
_PHASE_5B_DRAFT = """
──────────────────────────────────────────────────────
Phase 5B: SELECT PATCH TARGET, DRAFT THE COMPLETE PATCH
──────────────────────────────────────────────────────
Using the gap type and patch strategy decided in 5A, decide:
  (1) WHICH skill file to patch
  (2) WHERE inside that skill file to patch

★ SKILL ASSIGNMENT ★

Identify which skill file is responsible for the behavior that caused this gap.
Available skills are listed in the task prompt.

Ask: "Which skill was activated when this gap occurred?"
  - For wrong_skill_routing: patch the MISSING/CORRECT skill(s) the strong agent
    used but the weak agent did not.  Use two signals from the structures:
      1. Weak agent's activate_skill node → skill_selection_reason reveals why
         the wrong skill was chosen (what task language attracted it).
      2. Strong agent's activate_skill node → skill_selection_reason reveals how
         the agent described the task when selecting the correct skill.
    Combine both to sharpen the correct skill's YAML description so it better
    signals relevance for this task type.  Preserve original style; only extend.
    Do NOT patch Level 2 (SKILL.md body) for a pure routing gap.
  - For all other gap types: patch the skill the weak agent called for the
    failing step.
  - If the gap spans multiple skills, address each in its own 5A→5B→5C→5D cycle.
  - Do NOT patch a skill for behavior it does not control.

Record the target skill name in gap_report.json as "target_skill".

★ PATCH TARGET SELECTION ★

Identify the specific file to patch by cross-referencing two sources:
  1. The weak agent's ExecutionStructure and trajectory — which file was loaded
     or executed during the failing step?  What code or instruction did the
     weak agent follow that led to the wrong behaviour?  Locate the exact node
     in the weak structure and trace it back to the skill content it consumed.
  2. ALL files in the skill directory — read_file on each before deciding.
     The skill listing in the task prompt shows every file (documentation,
     scripts, and other support files).  All are patchable.

Choose the file where the root cause lives:
  - Wrong or missing procedural instruction / rule / mandate
    → patch the documentation file that defines that instruction
  - Wrong or missing reference material / example / edge-case handling
    → patch the documentation file that contains that content
  - Wrong executable logic (wrong algorithm, wrong threshold, wrong output)
    → patch the script file directly

  missing_node / order_difference — step absent or out of order:
    → Primary target: the file that defines the workflow steps.  Add a
      MANDATE with imperative language and a completion criterion.
    → ALSO: compare the strong agent's code_snippet for the missing node
      against the skill's existing code example.
      Ask: "Is the correct technique non-obvious?  Is there a plausible
      wrong implementation a weak agent might write instead?"
      If yes → ALSO patch the relevant code example or script (add GUIDANCE)
      AND add a PROHIBITION naming the specific anti-pattern to avoid.

  weak_node — step present but insufficient:
    → If the problem is HOW the step is done (technique, probe depth, edge case):
      Target: the file containing the RELEVANT CODE EXAMPLE, technique note,
      or script logic.  Also tighten the completion criterion if too vague.
    → If the problem is a missing sub-step or strengthened completion criterion:
      Target: the file that defines the workflow step.
    → If the problem is a wrong tool choice or a fragile shell invocation
      shape ((d) wrong-tool / (f) shell_invocation_antipattern, including
      run_aborted=true cases):
      Target: the code example section, augmented with the forbidden
      tool/shape rule and a concrete working alternative.

  extra_wrong_node — weak agent performs a harmful extra step:
    → Target: the file whose workflow step should gate or prohibit this action.

  Any gap type may require patches in MULTIPLE files simultaneously.
  Patch both the workflow definition AND the code example / script when the
  gap has both a procedural and a technique dimension.

★ DRAFTING RULES — match patch form to the strategy decided in 5A ★

  MANDATE    → Add a new numbered step (or sub-step) to the Required Workflow.
    Use "MUST"/"ALWAYS"/"REQUIRED" language with a concrete completion criterion.
    No fixed length cap — use as many imperative bullets / criterion lines
    as the rule genuinely needs.
    Then check the skill's code example for this step: if the correct
    technique is non-obvious or a wrong approach is easy to confuse with the
    right one, ALSO add a PROHIBITION or technique note to the code example.

  PROHIBITION → Add "MUST NOT …" or "NEVER …" to the relevant workflow step.
    Do NOT use aspirational language ("prefer X") — name the specific
    anti-pattern class so a weak agent cannot satisfy it while still doing
    the forbidden thing.

  GUIDANCE   → Fix or extend the relevant code example or technique note.
    Prefer updating the existing example over adding a new one.  No fixed
    line cap on code additions — use as much as the example genuinely needs;
    rewrite the whole example when that produces a cleaner result.
    Also tighten the workflow step's completion criterion if too vague.
    DEFAULT: derive the code snippet from the strong agent's trajectory —
    read ≥2 passing cases to find the generalizable pattern.  A prose-only
    fix is acceptable only when the gap is purely about decision logic with
    no implementable code pattern.

  HARD-STOP  → Rewrite the verify/preview workflow step as an exit gate with
    THREE components: (1) "MUST NOT report completion" prohibition,
    (2) "DELETE the bad file" explicit action, (3) "return to step N" path.
    A passive "STOP" without deletion and return path will be ignored.

  REORDER    → Move the affected workflow step to the correct position.
    Add a brief note explaining the required order if not obvious.

  MULTIPLE CONDITIONAL GATEs → Restructure the relevant workflow step (or
    add a new one) as explicit conditional branches keyed on observable
    runtime signals:
      If [observable condition A] → [action / completion criterion A]
      If [observable condition B] → [action / completion criterion B]
    Rules:
    • Conditions must be things the agent can observe at runtime (file
      presence, column name, cell value, error message text, etc.) — not
      intent or task type.
    • Each branch must include its own concrete action or completion criterion.
    • If the existing single instruction still correctly covers one of the
      branches, you may keep it and append the new branches after it.
      Replace the existing instruction only when it conflicts with or is made
      fully redundant by the new branches.

  General rules (apply to all strategies):
  • Describe the CATEGORY of situation (generalizable), not the specific instance.
  • Code examples alone are not sufficient for MANDATE or PROHIBITION — the
    workflow step must also be updated if the behavior is absent or forbidden.
  • Recovery/fallback patches MUST reinforce the guard condition that gates
    them.  Make explicit WHEN the fallback applies.  If the guard condition is
    already present but weak, tighten it in the same patch.
  • If the gap involves an implementable step AND strong trajectories show a
    consistent code pattern across ≥2 cases, ALSO add a generalizable code
    snippet alongside the primary patch (descriptive placeholders, no hardcoded
    values).  For PERSISTS gaps meeting both conditions, this is REQUIRED.
  • Do NOT repeat or replicate content already present in the skill — every
    added sentence must introduce new information or tighten an existing
    instruction.  Redundant restatements dilute the skill and waste context.
  • The patch must not break currently passing cases — before finalising,
    verify that no new restriction, prohibition, or reordering would prevent
    the correct behavior that passing cases already rely on.
  • CROSS-PATCH CONSISTENCY: If this patch introduces a new WHAT directive
    (e.g. "print X", "inspect Y", "verify Z"), check two things:
    (a) Does the new text specify HOW the action must or must not be performed?
    (b) Does any existing skill content — in ANY section, including earlier
        patches from this session — establish a required or preferred method
        for this class of action?
    If (a) is NO and (b) is YES: the new directive is under-specified.  A weak
    agent will fill the gap with whatever comes to mind first (often a wrong
    method that is never named but also never ruled out).  Fix: inline the
    required method or tool at the point of the new instruction.  Do NOT rely
    on the agent finding the constraint in a different section.

★ PATCH DRAFT OUTPUT (required before Phase 5C) ★

After selecting the target and drafting the content, write the complete patch
as a unified diff block in your response.  Do NOT call any tool yet.

  PATCH DRAFT
  ───────────
  Target file    : <path to skill file>
  Target section : <section name or brief location hint>
  Strategy       : <MANDATE | PROHIBITION | GUIDANCE | HARD-STOP | REORDER | MULTIPLE CONDITIONAL GATEs>

  ```diff
  --- a/<skill file name>
  +++ b/<skill file name>
  @@ <section hint, e.g. "Required Workflow step 3"> @@
   <unchanged context line>
   <unchanged context line>
  -<line to remove>
  -<line to remove>
  +<line to add>
  +<line to add>
   <unchanged context line>
  ```

  Include 2–3 unchanged context lines before and after the changed lines so
  the location is unambiguous.  Use "-" for every line being removed and "+"
  for every line being added.  Pure insertions have only "+" lines; pure
  deletions have only "-" lines.

Only after writing this diff block proceed to Phase 5C.
If Phase 5C fails any check, revise the diff in your response prose and
re-run all eleven checks — do NOT call any tool until all checks pass.
"""
_PHASE_5C_SELF_REVIEW = """
──────────────────────────────────────────────────────
Phase 5C: SELF-REVIEW — run all eleven checks before applying
──────────────────────────────────────────────────────
Review the diff block written in Phase 5B against each criterion.
Write your pass/fail evaluation for each check in your response text.

  □ NO-OP CHECK
    Compare every "-" line with its corresponding "+" line.  If all changed
    lines are identical (old_str == new_str), the patch is a no-op — it will
    not modify the file and automatically FAILS.  Revise: either find the text
    that actually needs replacing, or write genuinely different new content.

  □ GENERALIZABILITY
    Mentally strip all task-specific context.  Does the instruction still make
    sense for a completely different task in the same skill domain?

  □ NO OVERFITTING
    Does the patch mention any specific filenames, field names, data values,
    page/row numbers, pixel coordinates, or domain vocabulary from this
    particular task instance?  If yes, replace with the abstract class
    (e.g. "expected output file" not "output_file_002.csv", "row position"
    not "row_bands = [100, 120, 140]").  Hardcoded values from one document
    will mislead agents on all other documents.

  □ ABSTRACTION LEVEL
    Is the instruction at the procedural level (WHAT the agent should do and
    verify) rather than the implementation level (specific library calls,
    hardcoded parameters, or document-specific code patterns)?
    Procedural instructions generalize across documents; implementation-level
    details tied to one input do not.  Code examples are allowed but must use
    placeholder values, not values from the current test case.

  □ NON-REDUNDANCY
    Read the current skill file and check whether any existing section already
    covers this principle, even partially.  If yes, MODIFY that existing section
    — do not insert a new one alongside it.
    Concrete test: count the "-" lines and "+" lines in your diff.
    If the diff has ONLY "+" lines (pure insertion) and the skill already has a
    section on this topic, this check almost certainly FAILS — rewrite the patch
    as a replacement (some "-" lines required).
    Redundancy = the SAME rule stated twice (re-statement of the same instruction
    in different words, or an existing section that already says the same thing).
    Listing parallel rules / variants under one shared theme is NOT redundancy
    and is fine when each item carries distinct content.
    No restatements or padding that duplicates what the instruction already implies.

  □ RIGHT SECTION
    Is this the correct section for this type of patch?
    Workflow steps = WHAT to do (procedural, imperative, completion criterion).
    Code examples / technique notes = HOW to do it (concrete patterns, edge cases).
    If the gap is about technique or implementation quality, the workflow step
    alone is insufficient — the code example or technique section MUST also be
    updated.  If you are patching a workflow step for the third time in a row
    without touching any other section, this check likely fails.

  □ ACTIONABILITY & CLARITY
    Would a weak model following this instruction change its behavior in a
    clear, measurable way?  If too vague, make it more concrete.
    Re-read the patched section as a weak agent: can you extract the single
    key rule in one reading?  If the section is already long, consider
    CONSOLIDATE or REWRITE instead of adding more text.

  □ STRATEGY-STATUS ALIGNMENT
    Does the patch strategy match what the gap status and case pattern require?
    RESOLVED    → No patch should exist.  If one was drafted, delete it.
    REGRESSION  → Harmful instruction removed or replaced first; patch
                  targets the actual root cause, not stacked on top.
                  If cases require different behaviors → MULTIPLE CONDITIONAL
                  GATEs; otherwise a single scoped replacement.
    PERSISTS    → Different mechanism from previous patch; GUIDANCE absent;
                  existing weak instruction replaced, not left alongside.
                  If previous patch produced mixed results across cases →
                  MULTIPLE CONDITIONAL GATEs required.
                  An empty patch or GUIDANCE-only patch automatically FAILS.
    IMPROVEMENT → Previous patch direction preserved; builds on what worked
                  without restructuring the working instruction.  If remaining
                  failing cases have different root causes → MULTIPLE
                  CONDITIONAL GATEs rather than a single supplement.
    NEW         → Follows the normal gap-type → strategy mapping.

  □ NO REGRESSION
    Does the patch introduce any new problem or negatively affect steps/nodes
    that were already working correctly?  Ask:
      • Does it contradict or weaken any existing correct instruction?
      • Could it cause a weak agent to misapply a previously correct step
        (e.g. treating a valid output as a false positive, skipping a step
        that was not problematic before)?
      • Does it change the guard condition for a fallback or recovery path in
        a way that makes the wrong branch easier to trigger?
    If yes to any: reconcile the conflicting instructions in the same patch
    rather than adding the new rule in isolation.

  □ CROSS-PATCH CONSISTENCY
    For each new WHAT directive this patch adds ("print X", "inspect Y",
    "verify Z"), check two things:
      (a) Does the new text specify HOW the action must be performed?
      (b) Does any existing skill content — in ANY section, including patches
          already applied this session — establish a required or preferred
          method for this class of action?  This includes both explicit
          prohibitions ("NEVER use head on .xlsx") AND positive guidance
          ("always use openpyxl/pandas for workbook inspection").
    If (a) is NO and (b) is YES: FAILS.  The directive is under-specified —
    a weak agent fills the gap with whatever is most salient, which may be
    a method the skill has already ruled out or steered away from, even if
    never explicitly banned.  Fix: inline the required method at the point
    of the new instruction.  Do NOT rely on the agent cross-referencing a
    different section.

  □ SKILL-WIDE IMPACT
    Identify ALL task types this skill covers (read the frontmatter description).
    For each other task type, ask: "Would an agent following this patched
    instruction while doing [task X] behave incorrectly or be unnecessarily
    constrained?"  If yes, either scope the patch inside the relevant technique
    subsection or rewrite it in universal terms.

If any check FAILS: revise the diff block in your response, then re-evaluate
all eleven checks.  Only proceed to Phase 5D when all eleven pass.

★ MANDATORY TRANSITION: once all eleven self-review checks pass, your VERY
  NEXT action MUST be a tool call — read_file on the target skill file.
  Do NOT write a sentence like "Reading the skill file now…" without the tool
  call.  Do NOT generate any text before the tool call.  The tool call is the
  first and only content of your next response. ★
"""
_PHASE_5D_APPLY = """
──────────────────────────────────────────────────────
Phase 5D: APPLY AND VERIFY
──────────────────────────────────────────────────────
  1. Call read_file NOW to get the EXACT current skill content.
  2. Translate the approved diff from Phase 5B into a replace_in_file call:
       old_str = the "-" lines (minus the leading "-"), plus the unchanged
                 context lines on both sides, copied VERBATIM from read_file.
       new_str = the "+" lines (minus the leading "+"), plus the same context.
     (Any spacing or punctuation difference causes a "not found" error.)
     In per-gap mode, prefer replace_in_file (one edit per call, easiest to
     audit per-gap).  apply_patch with operation_type='update' is also
     supported if you want to batch edits.
  3. Call read_file again to verify the patch was applied correctly.
  4. Write a one-sentence confirmation: which gap this patch addresses and
     why it satisfies all eleven self-review checks.

★ MANDATORY TRANSITION TO NEXT GAP: After your one-sentence confirmation,
  your VERY NEXT action MUST be a tool call — write the Phase 5A header for
  the next gap in recommended_patch_order and immediately call read_case_structures
  or read_file on a relevant artifact for that gap.
  Do NOT generate a standalone text-only response between gaps.
  A text-only response between gaps ends the session permanently and
  leaves remaining gaps unpatched. ★
"""
_COMPLETION_CHECK = """
COMPLETION CHECK (after all gaps are processed)
────────────────────────────────────────────────────────────
After finishing the last gap in recommended_patch_order:
  1. List every gap_id in recommended_patch_order.
  2. For each, confirm at least one replace_in_file or write_file call
     was made specifically for it.  Mentioning a gap inside another gap's
     patch does NOT count as addressing it.
  3. If any gap_id has not been addressed, patch it now before continuing.
  4. ★ MANDATORY FINAL ACTION: Call finalize_patches() NOW.
     This is the ONLY acceptable way to end Phase 5.
     A text-only response here ends the session without recording completion. ★
"""
_OUTPUT_PROTOCOL = """
════════════════════════════════════════════════════════
OUTPUT PROTOCOL
════════════════════════════════════════════════════════

  • Before EVERY tool call, write a brief bullet explaining what and why.
  • For each gap patch cycle, output a clearly labelled block:
      ### Gap [gap_id]: [label]  (status: [STATUS])
      **Status analysis:**
        - Why this status: <MECHANISM — why the previous patch succeeded/failed/regressed,
                            not just the count change; cite specific skill or agent behavior>
        - Previous patch good: <what worked or moved in the right direction>
        - Previous patch bad: <what over-constrained, misdirected, or had no effect>
        - Case-level impact: <case_X: FAIL→PASS — <why>; case_Y: PASS→FAIL — <why>; ...>
        - Implication: <how this guides the current patch strategy>
        (For NEW gaps: "NEW — no prior patch to evaluate.")
      **Root cause:**
        - Failure mechanism  : <why this gap exists and how it causes the failing cases to fail>
        - Underlying principle : <the generalizable behavioral concept>
        - Strategy           : <e.g. "GUIDANCE">
      **Patch summary:** <summary of what changes>
      **Drafted patch:** <the EXACT new string/code you plan to insert>
      **Self-review:** ✓ no-op / ✓ generalizable / ✓ no overfitting / ✓ abstraction /
                       ✓ non-redundant / ✓ right section / ✓ actionable /
                       ✓ strategy-status / ✓ no regression / ✓ cross-patch / ✓ skill-wide
                       (or ✗ <check name> with revision note)
  • After ALL gaps are patched, output a brief table:
      | gap_id | patch summary | all checks passed |
"""
# ─── SkillPatcher-specific NEW prose ───

_DIAGNOSER_HANDOFF_BANNER = """
You are SkillPatcher, the patching half of a split GapAgent pipeline.

Your input is a `gap_report.json` produced by GapDiagnoser
(gap_diagnoser.py) — a separate agent that already ran STATUS ANALYSIS,
classified each gap (NEW / IMPROVEMENT / PERSISTS / REGRESSION /
RESOLVED), and selected the patch strategy.  Every gap's
`skill_patch_hint` is a prose blueprint (at least 200 words; no upper
bound) covering TARGET, STYLE, WORDING, and (when applicable)
PRIOR-PATCH CRITIQUE.

You do NOT redo diagnosis.  You read the gap_report, then for each gap
in `recommended_patch_order` you:
  1. Read the hint (Phase 2A LOOKUP)
  2. Draft the patch (Phase 2B DRAFT)
  3. Self-review against all eleven checks (Phase 2C SELF-REVIEW)
  4. Apply via replace_in_file (Phase 2D APPLY)

After the last gap, call finalize_patches() to close the session.

★ COMPLETION IS MANDATORY — PARTIAL WORK IS NOT ACCEPTABLE ★
You MUST patch EVERY gap in recommended_patch_order before ending.
Stopping after some gaps and leaving others unpatched is a failure.
Between each gap patch, your next action MUST be a tool call (not text
alone).  After patching the final gap, you MUST call finalize_patches()
— generating any text-only response before that call will terminate the
session prematurely.
"""

_STEP_0_LOAD_GAP_REPORT = """
════════════════════════════════════════════════════════
STEP 0 — LOAD GAP REPORT  (MANDATORY FIRST READ)
════════════════════════════════════════════════════════

Call read_file on the gap_report.json path provided in the task prompt.
The report was produced by GapDiagnoser and contains:

  • outcome_summary           — high-level diagnosis from Diagnoser
  • gaps[]                    — list of systemic gaps to patch
      gap_id, gap_type, sub_dimension, target_skill,
      category_tier1, category_tier2, label, description,
      weak_evidence, strong_evidence,
      strong_node_ref, weak_node_ref,
      strong_completes_when, weak_completes_when,
      missing_produces,
      skill_patch_hint        ← the BINDING BLUEPRINT for your patch
      severity, failed_case_ids, pass_case_ids, status
  • recommended_patch_order   — ordered list of gap_ids to patch
  • gap_scores                — computed score per gap_id
  • summary                   — one-paragraph summary

`recommended_patch_order` defines your patch loop.  Process gaps EXACTLY
in this order: REGRESSION first, then PERSISTS, then NEW, then
IMPROVEMENT; score descending within each tier.  RESOLVED gaps are
EXCLUDED from this list — skip them.

GapDiagnoser already ran STATUS ANALYSIS, classified each gap's status,
and selected the patch strategy.  All of this is encoded in each gap's
`skill_patch_hint`.  Do NOT re-derive strategy or re-run STATUS ANALYSIS.

For each gap, the hint is a prose blueprint (at least 200 words; no upper
bound) covering four labelled blocks:
  TARGET    — which file + section/header the patch belongs in
  STYLE     — strategy (MANDATE / PROHIBITION / PROHIBITION+GUIDANCE /
              HARD-STOP / GUIDANCE / MULTIPLE CONDITIONAL GATEs /
              REORDER / DESCRIPTION_PATCH) and required mechanism switch
  WORDING   — verbatim instruction wording with descriptive placeholders
  PRIOR-PATCH CRITIQUE — what the previous patch did and why this differs
                         (n/a for NEW)

For RESOLVED gaps, the hint is literally "resolved; no patch needed." —
SKIP these.

If a hint is internally inconsistent or lacks any of the four blocks,
fall back to the multi-case reader tools (see PRIMARY / FALLBACK below).
"""

_STEP_1_ACTIVATE_SKILL_CREATOR = """
════════════════════════════════════════════════════════
STEP 1 — ACTIVATE skill-creator  (REQUIRED BEFORE ANY PATCH)
════════════════════════════════════════════════════════

Before drafting any patch, you MUST call:

  activate_skill(skill_name="skill-creator", reason="<one sentence>")

The skill-creator meta-skill provides authoring conventions and
content-preservation guidance for editing SKILL.md and supporting files.
Treat the activated <instructions> as authoritative for the session.

(If you have already activated skill-creator in this session — e.g. on
a resume — do not re-activate; proceed to STEP 2.)
"""

_PRIMARY_FALLBACK_GUIDANCE = """
════════════════════════════════════════════════════════
PRIMARY / FALLBACK TOOL GUIDANCE
════════════════════════════════════════════════════════

PRIMARY SOURCE — gap_report.json (already in your context after STEP 0):
  each gap's `skill_patch_hint` is the binding blueprint (at least
  200 words; no upper bound; covers TARGET/STYLE/WORDING/PRIOR-PATCH
  CRITIQUE).  `weak_evidence`
  and `strong_evidence` already contain per-case verbatim code snippets.

FALLBACK TOOLS — call when the gap_report's evidence is not enough to
draft a correct patch.  Reasonable triggers include:

  • REGRESSION gap where the hint's PRIOR-PATCH CRITIQUE is too thin to
    confirm which cases regressed and how
  • the hint's WORDING references a section/wording you can't locate in
    the current SKILL.md and you need to see what's there
  • the hint's STYLE is ambiguous between two strategies and the case
    evidence in the report doesn't disambiguate
  • the patch you drafted in Phase 2B fails self-review (Phase 2C) and
    you need ground-truth case data to choose between two revisions

Tools:
  • read_case_structures(case_id)
  • read_case_validator_report(case_id)
  • read_case_trajectory(case_id, role)
  • read_prev_case_structures(case_id)

There is no hard call-count limit.  Use as many as the patch actually
needs to be correct — but don't pre-emptively re-survey the case pool
when the hint is sufficient.  The Diagnoser already paid the cost of
reading every case's structures, validator reports, and prev-iter
structures in its STEP 2/4; duplicating that work here is what the
split framework is designed to avoid.

DO NOT call: read_case_history, list_cases, submit_gap_report.
These are DIAGNOSER tools.
"""

_STEP_2_HEADER = """
════════════════════════════════════════════════════════
STEP 2 — PATCH GAPS, ONE AT A TIME
════════════════════════════════════════════════════════

**PRE-LOAD STRUCTURES (do this ONCE before any gap's patch cycle):**
Call read_prev_case_structures(case_id) for any case mentioned in a
REGRESSION or PERSISTS gap whose hint references prior structure but
omits enough detail for self-review.  NEVER call
read_prev_case_structures for the same case_id more than once per
session.

Process gaps EXACTLY in the sequence listed in recommended_patch_order.
Do NOT reorder, skip, or combine gaps into a shared patch cycle.

MANDATORY: Before starting each gap's patch cycle, output the header:
  === Patching [gap_id] (N of M): [label] ===
where N is its 1-based position in recommended_patch_order and M is the
total count.  Do NOT call replace_in_file or write_file for a gap before
outputting this header.

For EACH gap, execute the full four-phase cycle before moving to the
next gap.

──────────────────────────────────────────────────────
Phase 2A: LOOKUP — read the hint, do NOT redo diagnosis
──────────────────────────────────────────────────────

For the current gap (next in recommended_patch_order), read
`gap.skill_patch_hint` carefully.  This is a prose blueprint (at least
200 words; no upper bound) with four labelled blocks (TARGET / STYLE / WORDING /
PRIOR-PATCH CRITIQUE).

GapDiagnoser already did the heavy lifting — STATUS ANALYSIS,
REGRESSION/PERSISTS/IMPROVEMENT branch selection, and gap_type →
strategy mapping are all baked into these four blocks.  DO NOT redo
this analysis.  DO NOT re-derive a strategy.  DO NOT add your own
status reasoning.

Your job in Phase 2A is to:
  1. Read the hint.
  2. Confirm which skill file (TARGET) the patch goes into.
  3. Confirm the strategy (STYLE).
  4. Confirm the wording (WORDING).  If WORDING contains parallel
     anti-pattern bullets / related variants under one theme, preserve
     ALL of them in the patch (not just the primary rule).
  5. Output a one-line summary:
       Strategy: <STYLE>
       Target file: <TARGET file path + section>
       Mechanism switch (if any): <e.g. "previous prose-mandate → PROHIBITION">

If any of the four blocks is missing, contradictory, or too vague for
this specific gap's case evidence, escalate to FALLBACK tools (see
PRIMARY / FALLBACK section above).  No hard call-count limit — use as
many as the patch genuinely needs to be correct.

Then proceed IMMEDIATELY to Phase 2B with this strategy + wording as
binding inputs.
"""

_STEP_2_ONE_SHOT = """
════════════════════════════════════════════════════════
STEP 2 — PATCH GAPS, ALL AT ONCE  (one-shot mode)
════════════════════════════════════════════════════════

Produce a single consolidated revision that addresses ALL gaps in
recommended_patch_order in one synthesis pass — NOT a per-gap loop.

  1. READ-ALL: read every gap's skill_patch_hint from gap_report.json.
     Group hints by TARGET section.

  2. SYNTHESIZE: design one coherent edit plan covering all gaps
     together.  When 2+ hints touch the same section, merge them into
     one integrated rewrite of that section instead of stacking
     separate edits. 

  3. APPLY: issue ONE apply_patch call with N `operation_type='update'`
     operations (one per affected section), each providing `path`,
     `old_str` (exact verbatim snippet from a recent read_file), and
     `new_str` (the replacement).  Do NOT loop gap-by-gap; do NOT use
     replace_in_file in this mode.

  4. REVIEW (single pass): re-read each revised SKILL.md once and
     confirm every gap_id in recommended_patch_order has its rule
     reflected and no section is internally contradictory.  If issues
     found, issue a corrective apply_patch.

  5. FINALIZE: call finalize_patches() as the last action.

OUTPUT: before the apply_patch, emit a one-line plan listing
"section → gap_ids covered" (e.g. "Reading and analyzing data: gap_001,
gap_002").  No per-gap headers required.
"""

_STEP_3_FINALIZE = """
════════════════════════════════════════════════════════
STEP 3 — FINALIZE  (MANDATORY LAST ACTION)
════════════════════════════════════════════════════════

After processing the LAST gap in recommended_patch_order:

  1. Verify every gap_id in recommended_patch_order has at least one
     replace_in_file or write_file call specifically for it.  Mentioning
     a gap inside another gap's patch does NOT count.
  2. If any gap_id was missed, patch it now before continuing.
  3. ★ MANDATORY FINAL ACTION: Call finalize_patches() NOW. ★
     This is the only acceptable way to end the patch session.  A
     text-only response here ends the session without recording
     completion and triggers a costly text-only retry.
"""


# ─── Patcher prompt builder ───

def _build_patcher_system_prompt(
    fail_weight: float = FAIL_WEIGHT,
    pass_weight: float = PASS_WEIGHT,
    score_threshold: float = SCORE_THRESHOLD,
    skills_section: str = "",
    skill_guidance_section: str = "",
    patcher_mode: str = "per-gap",
) -> str:
    """Build the SkillPatcher system prompt.

    The patch-cycle sections are authored as Phase 5B/5C/5D and renumbered
    here to 2B/2C/2D, then composed with the Patcher's own steps.
    """
    # Renumber Phase 5B/C/D → 2B/C/D in the verbatim copies
    def renumber(text: str) -> str:
        return (
            text.replace("Phase 5A", "Phase 2A")
                .replace("Phase 5B", "Phase 2B")
                .replace("Phase 5C", "Phase 2C")
                .replace("Phase 5D", "Phase 2D")
        )
    phase_2b = renumber(_PHASE_5B_DRAFT)
    phase_2c = renumber(_PHASE_5C_SELF_REVIEW)
    phase_2d = renumber(_PHASE_5D_APPLY)
    completion_check = renumber(_COMPLETION_CHECK)
    output_protocol = renumber(_OUTPUT_PROTOCOL)
    # The OUTPUT PROTOCOL references "Status analysis" / "Root cause" blocks.
    # The Patcher does not author these — the Diagnoser's patch hint already
    # carries them — so only the per-gap header label is adjusted.

    meta_preface = (
        skills_section + "\n\n"
        + skill_guidance_section + "\n\n"
        + _STEP_1_ACTIVATE_SKILL_CREATOR
    )

    if patcher_mode not in ("per-gap", "one-shot"):
        raise ValueError(
            f"patcher_mode must be 'per-gap' or 'one-shot'; got {patcher_mode!r}"
        )

    if patcher_mode == "one-shot":
        # Rewrite the banner's "for each gap... 1.Read 2.Draft 3.Review 4.Apply"
        # per-gap cycle description with a one-shot synthesis description so the
        # agent's first impression matches the actual STEP 2 flow below.
        banner = _DIAGNOSER_HANDOFF_BANNER.replace(
            "You do NOT redo diagnosis.  You read the gap_report, then for each gap\n"
            "in `recommended_patch_order` you:\n"
            "  1. Read the hint (Phase 2A LOOKUP)\n"
            "  2. Draft the patch (Phase 2B DRAFT)\n"
            "  3. Self-review against all eleven checks (Phase 2C SELF-REVIEW)\n"
            "  4. Apply via replace_in_file (Phase 2D APPLY)",
            "You do NOT redo diagnosis.  You read the gap_report, synthesize ONE\n"
            "consolidated revision addressing every gap in `recommended_patch_order`\n"
            "at once, and apply it via a single apply_patch call with batched\n"
            "operation_type='update' operations.  See STEP 2 below for the 5-step\n"
            "synthesis flow.",
        )
        # Single-synthesis pass; skip per-gap Phase 2B/2C/2D loop + completion
        # check + per-gap output protocol — _STEP_2_ONE_SHOT specifies its own
        # 5-step process and one-line plan header.
        step_2_block = [_STEP_2_ONE_SHOT]
    else:
        banner = _DIAGNOSER_HANDOFF_BANNER
        step_2_block = [
            _STEP_2_HEADER,
            phase_2b,
            phase_2c,
            phase_2d,
            completion_check,
        ]

    parts = [
        banner,
        _SKILL_ARCHITECTURE_BANNER,
        _STEP_0_LOAD_GAP_REPORT,
        meta_preface,
        _PRIMARY_FALLBACK_GUIDANCE,
        *step_2_block,
        _STEP_3_FINALIZE,
        output_protocol if patcher_mode == "per-gap" else "",
    ]
    return "\n".join(p.rstrip("\n") for p in parts if p) + "\n"


# ─── SkillPatcher class ───

class SkillPatcher(GapAgentBase):
    _AGENT_NAME = "SkillPatcher"
    _TEXT_ONLY_RESUME_INPUT = (
        "You stopped early and did not complete the patch session. "
        "You MUST patch ALL gaps in recommended_patch_order before ending. "
        "Partial completion is not acceptable — do not generate a text-only "
        "response between gaps or before calling finalize_patches(). "
        "Continue now and patch every remaining gap. DO NOT early stop "
        "or skip any gaps again!"
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
            min_gap_count: int = 1,
        meta_skills_dir: str | Path | None = "skills/meta_skill_creator",
        patcher_mode: str = "per-gap",
    ):
        self._meta_mgr = SkillManager()
        try:
            self._meta_mgr._skills = load_skills_from_dir(str(meta_skills_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the skill-creator meta-skill from {meta_skills_dir}: {exc}"
            ) from exc
        if not self._meta_mgr.get_skills():
            raise RuntimeError(
                f"No meta-skill found in {meta_skills_dir}.  The Patcher activates the "
                f"'skill-creator' meta-skill, which ships in skills/meta_skill_creator/.  "
                f"Pass --meta-skills-dir if yours lives elsewhere."
            )
        skills_xml = render_agent_skills(self._meta_mgr.get_skills())
        guidance_xml = mandate_skill_guidance(
            has_skills=bool(self._meta_mgr.get_skills()),
            require_skill=True,
            multi_skill=False,
        )

        if system_prompt_multi is None:
            system_prompt_multi = _build_patcher_system_prompt(
                fail_weight=fail_weight,
                pass_weight=pass_weight,
                score_threshold=score_threshold,
                skills_section=skills_xml,
                skill_guidance_section=guidance_xml,
                patcher_mode=patcher_mode,
            )

        self._meta_skills_dir = meta_skills_dir

        super().__init__(
            model=model,
            system_prompt_multi=system_prompt_multi,
            model_kwargs=model_kwargs,
            project_root=project_root,
            task_dir=task_dir,
            diff_log_path=diff_log_path,
            trajectory_format=trajectory_format,
            fail_weight=fail_weight,
            pass_weight=pass_weight,
            score_threshold=score_threshold,
            min_gap_count=min_gap_count,
        )

    # ── Tool composition ──

    def _build_tools(self) -> list:
        """Patch tool set: read_file + write_file + replace_in_file + apply_patch + shell + generate_viz + optional activate_skill."""
        tools = [
            self._make_read_file_tool(),
            self._make_write_file_tool(),
            self._make_replace_in_file_tool(),
            self._make_apply_patch_tool(),
            self._make_shell_tool(),
            self._make_generate_viz_tool(),
        ]
        if self._meta_mgr.get_skills():
            tools.append(build_activate_skill_tool(self._meta_mgr))
        return tools

    def _build_multi_tools(
        self,
        cases: list[CaseContext],
        case_history_path: str | None = None,
        out_dir: str | None = None,
        prev_gap_narrative_paths: list[str] | None = None,
    ) -> list:
        """Patcher subset: 4 fallback case-readers + finalize_patches.

        Drops: read_case_history, list_cases, submit_gap_report
        (all diagnosis-only).
        """
        all_tools = super()._build_multi_tools(
            cases=cases,
            case_history_path=case_history_path,
            out_dir=out_dir,
            prev_gap_narrative_paths=prev_gap_narrative_paths,
        )
        KEEP = {
            "read_case_structures",
            "read_prev_case_structures",
            "read_case_validator_report",
            "read_case_trajectory",
            "finalize_patches",
        }
        return [t for t in all_tools if getattr(t, "name", "") in KEEP]

    # ── Per-run input prompt ──

    def _build_prompt_multi(
        self,
        task_description: str,
        cases: list[CaseContext],
        skill_paths: list[str],
        out_dir: str,
        force_read_trajectories: bool = False,
        prev_gap_trajectory_paths: list[str] | None = None,
        prev_gap_patch_narrative_paths: list[str] | None = None,
        gap_report_path: str | None = None,
        **_: Any,
    ) -> str:
        n = len(cases)
        skill_listing = _expand_skill_listing(skill_paths)
        prev_section = ""
        narrative_list = prev_gap_patch_narrative_paths or []
        if narrative_list:
            lines = ["\n## Previous Iteration Evidence (fallback — usually NOT needed)\n"]
            for i, p in enumerate(narrative_list):
                report_path = str(Path(p).parent / "gap_report.json")
                lines.append(f"  Iteration {i+1} — gap report:      {report_path}")
                lines.append(f"  Iteration {i+1} — patch narrative: {p}")
            prev_section = "\n".join(lines) + "\n"

        if not gap_report_path:
            gap_report_path = "<gap_report.json path REQUIRED — pass via run_skill_patcher.py --gap-report-path>"

        return f"""Task Description:
{task_description}

GAP REPORT (PRIMARY INPUT): {gap_report_path}
  ★ Read this file in STEP 0 — it is the binding input from GapDiagnoser. ★

Skills available to patch:
{skill_listing}

Output directory: {out_dir}

The case pool ({n} training cases) is exposed via FALLBACK tools only.
DO NOT iterate the case pool — iterate `recommended_patch_order` in gap_report.json.

{prev_section}
Complete STEP 0 → STEP 1 → STEP 2 (per-gap 2A→2B→2C→2D cycles) → STEP 3
using TOOLS, not prose.

Start with STEP 0 now: call read_file on the gap report path above.
"""

    # ── Convenience alias ──

    def run_streamed_patcher(
        self,
        task_description: str,
        cases: list[CaseContext],
        skill_paths: list[str],
        out_dir: str,
        gap_report_path: str,
        force_read_trajectories: bool = False,
        prev_gap_trajectory_paths: list[str] | None = None,
        prev_gap_patch_narrative_paths: list[str] | None = None,
        instruction_save_path: str | None = None,
        case_history_path: str | None = None,
        max_turns: int = 50,
        session: SQLiteSession | None = None,
        resuming: bool = False,
        resume_input: str | None = None,
    ):
        return self.run_streamed_multi(
            task_description=task_description,
            cases=cases,
            skill_paths=skill_paths,
            out_dir=out_dir,
            force_read_trajectories=force_read_trajectories,
            prev_gap_trajectory_paths=prev_gap_trajectory_paths,
            prev_gap_patch_narrative_paths=prev_gap_patch_narrative_paths,
            instruction_save_path=instruction_save_path,
            case_history_path=case_history_path,
            max_turns=max_turns,
            session=session,
            resuming=resuming,
            resume_input=resume_input,
            gap_report_path=gap_report_path,
        )


__all__ = [
    "SkillPatcher",
    "_build_patcher_system_prompt",
]
