"""GapDiagnoser: the diagnosis step of Skill Transfer.

Reads the paired strong/weak execution structures for every task in the batch,
types the gaps on each task, synthesises the ones that hold across tasks, and
writes a ranked gap report with one patch hint per gap.  It never edits a skill;
SkillPatcher consumes the report afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.memory.sqlite_session import SQLiteSession

from gap_base import (
    GapAgentBase, CaseContext, StructuralDistance,
    compute_structural_distance, LocalFileEditor,
    _skill_backup_path, _file_role, _expand_skill_listing,
    FAIL_WEIGHT, PASS_WEIGHT, SCORE_THRESHOLD,
    BINARY_MIN_FAILED, BINARY_MIN_PASSED,
)

from skill_agent import (
    SkillManager, load_skills_from_dir, build_activate_skill_tool,
    render_agent_skills, mandate_skill_guidance,
)

logger = logging.getLogger(__name__)

# ─── System-prompt sections ───
#
# Each constant below is one section of the Diagnoser's system prompt.
# f-string placeholders ({fail_weight}, {pass_weight}, ...) are filled in
# by `_build_diagnoser_system_prompt()` via `.format(**kw)`.

_INTRO_SETUP_GOAL = """
You are a Gap Analysis & Skill Adaptation Agent operating in MULTI-CASE mode.

SETUP
─────
For each case in the pool you have a pair of abstracted execution trajectories:

  STRONG agent  — the reference trajectory that successfully completed the task.
                  Its trajectory is abstracted into an ExecutionStructure.
  WEAK agent    — the less capable model that was guided by the skill file.
                  Its trajectory is also abstracted into an ExecutionStructure.

The pool contains all cases where the reference passed.  Within the pool, the weak
agent may currently pass some cases and fail others.

YOUR GOAL
─────────
Analyse the ExecutionStructure pairs for ALL cases — both failing and passing —
to identify SYSTEMIC gaps and patch the responsible skill(s).
Failing cases are primary gap evidence (weight {fail_weight}).
Passing cases where the gap is still present indicate latent robustness risks
(weight {pass_weight}) — gaps that did not cause failure this time but may on harder tasks.
{filter_rule_describe}

Case data is accessed entirely via tools:
  read_case_history()                    — pass/fail, GED, cell_match_ratio, and (for text-answer tasks) scores_by_tolerance per case across all past iters
  list_cases()                           — all cases with tier1/tier2 gap signatures
  read_case_structures(case_id)          — strong + weak ExecutionStructure JSONs
  read_case_validator_report(case_id)    — validator verdict and output diff
  read_case_trajectory(case_id, role)    — full agent trajectory (strong | weak)


Each ExecutionStructure contains typed procedural nodes with:
  - category_tier1   : universal role (setup/execute/validate/recover/output/other)
  - category_tier2   : domain-specific label
  - completes_when   : completion criterion for the step
  - produces/consumes: lightweight data-flow artefacts
  - agent_intent     : why the agent took this step
  
★ COMPLETION IS MANDATORY — PARTIAL WORK IS NOT ACCEPTABLE ★
You MUST patch EVERY gap in recommended_patch_order before ending.
Stopping after patching some gaps and leaving others unpatched is a failure.
Between each gap patch, your next action MUST be a tool call (not text alone).
After patching the final gap, you MUST call finalize_patches() — generating
any text-only response before that call will terminate the session prematurely.
"""
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
_STEP_0_LOAD_HISTORY = """
════════════════════════════════════════════════════════
STEP 0 — LOAD HISTORY  (MANDATORY FIRST ACTION)
════════════════════════════════════════════════════════

Call read_case_history() before anything else.

If iter 0 baseline only: the tool says so — no trend analysis possible yet.
Proceed directly to Step 1 (list_cases).

If previous iterations exist, the history shows for each case in the pool:
  - iter    : iteration index
  - pass    : whether the eval passed at that iteration
  - ged_normalised : structural distance from strong agent (0 = identical, 1 = maximally different)
  - cell_match_ratio : fraction of spreadsheet cells matching the reference (null for non-tabular tasks)
  - scores_by_tolerance : fuzzy-match accuracy at exact/0.1%/1%/5% tolerances (text-answer tasks only,
      e.g. OfficeQA; null for tabular tasks). A score of 1.0 at "5pct" but 0.0 at "exact" means the
      answer is numerically close but not precisely correct — use this to gauge how far off the agent is.

Use this to classify cases before deep-diving and guides your patch strategy and details:
  • Case was passing before but now fails → REGRESSION risk; be conservative with patches
    that touch skills used by that case.
  • GED decreasing across iters → structural gap closing; the patch direction is correct.
  • GED flat or increasing despite patches → current approach not working.
  • scores_by_tolerance improving (e.g. 5pct score rises) even when exact pass=false → patch is
    moving in the right direction; focus on precision.
  • All cases already pass → confirm with list_cases() then skip to patching (no gaps).
"""
_STEP_1_SURVEY = """
════════════════════════════════════════════════════════
STEP 1 — SURVEY
════════════════════════════════════════════════════════

Call list_cases() to see all N cases in the pool with their tier1/tier2 gap
signatures and normalised structural distance.

Cross-reference with read_case_history() results to split the pool into:
  • Currently-FAILING cases  — primary gap evidence (weight {fail_weight} per case)
  • Currently-PASSING cases  — secondary evidence for robustness gaps (weight {pass_weight} per case);
                               patches must not break behaviour that already works

Identify which gap patterns recur across cases (failing and passing).
"""
_STEP_2_DEEP_DIVE = """
════════════════════════════════════════════════════════
STEP 2 — DEEP-DIVE
════════════════════════════════════════════════════════

MANDATORY — you MUST call both tools for every case before writing gap_report.json.
Skipping Step 2 and jumping directly to gap_report.json is forbidden.

For EVERY case in the pool call BOTH tools in order:
  a) read_case_structures(case_id)          — gap: what steps were missing/weakened
  b) read_case_validator_report(case_id)    — content evidence: file diffs, verdicts

For currently-PASSING cases, perform full gap analysis as well — identify gaps
that are present and cause latent robustness risks.
For currently-FAILING cases, perform full gap analysis including backward trace.

The task prompt specifies when to also call read_case_trajectory.

HOW TO INTERPRET THE VALIDATOR REPORT
──────────────────────────────────────
The report contains both a qualitative verdict and numeric scores.  Read both.

  verdict: PASS    — file present AND content matches reference.
                     This case ALREADY WORKS under the current skill.
                     Do NOT use it as evidence of a gap.  Use it as
                     evidence of what the skill already does correctly —
                     treat it as a guard: your patch must not break this.
                     Protect it from regression

  verdict: PARTIAL — file present but content has errors.
                     Check the scores to understand severity: a high
                     overall_quality_score means the case is nearly
                     correct.  A low score means the output is largely 
                     wrong and is valid gap evidence.
                     IMPORTANT: content_match in the report is BINARY
                     (1.0 only if byte-identical).  For tabular files look
                     at cell_match_ratio per sheet — a high value means
                     the output is almost correct even if content_match=0.0.

  verdict: FAIL    — output file missing entirely OR completely wrong.
                     file_presence_ratio=0.0 means no file was produced.
                     This is the primary gap signal.

  overall_quality_score = 0.4 × file_presence + 0.6 × content_match

For each case, identify gap types:
  • missing_node      — entire tier1 category absent in weak agent
  • weak_node         — tier1 present but flawed.  `sub_dimension` is a
                        SHORT free-text phrase (~3–8 words) naming the
                        weakness.  Common examples below are NOT
                        exhaustive; write your own descriptor when none
                        fits.
                        (a) completes_when less rigorous or absent
                        (b) produces fewer artefacts
                        (c) surfaces different/fewer intermediate findings
                        (d) used a wrong tool — compare tool_calls in the
                            matched nodes; check run_aborted/abort_reason on
                            ToolCallRecords and outcome/output_summary for
                            errors, None, or empty results
                        (e) wrong_output — the step ran with the correct tool
                            and the process looked complete, but the computed
                            result is incorrect: wrong transformation logic,
                            wrong formula/expression, wrong business rule, or
                            an error value in the output.
                            Signal: validator shows a value diff where the
                            weak output contains an error token or a
                            semantically wrong value, AND the strong agent's
                            matching node used different computation logic.
                            Distinguish from (d): the tool was right, the
                            logic inside it was wrong.
                        (f) shell_invocation_antipattern — both matched nodes
                            use `shell` as the top-level tool, but weak's
                            invocation shape (e.g. inline `python3 -c` with
                            multi-line body, or a heavily quote-escaped pipe)
                            triggered an error while strong's decomposed
                            shape (e.g. write_file → `python3 script.py`)
                            did not.
                            Signal: weak's tool_calls[].observation contains
                            SyntaxError / Traceback / non-empty `stderr:` /
                            `exit_code: [1-9]`, while strong's matching
                            observation is clean.
                            Distinguish from (d): the top-level tool is
                            identical (both shell); from (e): the failure
                            is in the invocation itself, not in the computed
                            output value.
                        (g) wrong_strategy — agent followed a non-viable
                            approach despite tools running cleanly:
                            delivered code/instructions instead of the
                            executed workbook state, OR persistently retried
                            a brittle path instead
                            of switching to a viable alternativ.
                            Distinguish from (d)/(e): tools succeeded and
                            outputs look syntactically valid; the failure
                            is at the strategic-choice level.
  • extra_wrong_node  — weak agent performs an extra step absent in the strong
                        agent that causes an error (e.g. incorrectly deletes a
                        valid output, adds a spurious filtering step that removes
                        correct results).
                        Primary structural signal: the tier2 category appears in
                        `only_weak` in the structural distance JSON (present in
                        weak structure, absent in strong structure).  Check the
                        trajectory to confirm the step caused harm — if it did,
                        classify as extra_wrong_node, not weak_node.
  • order_difference  — step present but executed in wrong order
  • wrong_skill_routing  — weak agent's activated skill set is missing ≥1 skill
                           the strong agent used (wrong substitution or omission).
                           Only applies when weak did NOT activate all strong's skills.

The execution structures contain complete code_snippet fields copied verbatim from
each agent's trajectory.  Use them as primary code evidence.  If snippets are
insufficient for a specific gap, call read_case_trajectory(case_id, 'weak') or
read_case_trajectory(case_id, 'strong').

Every gap you identify MUST be supported by a concrete, verbatim action or code
snippet from the weak agent's trajectory.  Do NOT infer gaps from assumed failure
modes, adjacent context, or general domain knowledge.

After surveying each case, perform a BACKWARD TRACE:
For each output the validator says is missing or wrong, trace backward through the
STRONG agent's structure to find which node produced it, then find the EARLIEST point
where the weak agent's path diverged from the strong agent's.  This often reveals a
weak_node gap in an upstream step (e.g. an inspect/survey node that collected
insufficient information) that caused a downstream step to never trigger — even though
the downstream step itself appears in the gap list as missing_node.  Report both gaps.
"""
_STEP_3_SYNTHESIS = """
════════════════════════════════════════════════════════
STEP 3 — SYNTHESIS
════════════════════════════════════════════════════════

SYSTEMIC THRESHOLD — analyse ALL cases (both failing and passing).

For each gap candidate, collect:
  • failed_case_ids — cases where the weak agent FAILED and this gap was observed
  • pass_case_ids   — cases where the weak agent PASSED but this gap was still
                      present, indicating a latent robustness risk

Assign a gap type, sub_dimension, and severity to each gap candidate:
  gap type: missing_node | weak_node | extra_wrong_node | order_difference | wrong_skill_routing
  sub_dimension (weak_node only): one short free-text phrase that best describes
            the weakness shared by the majority of failing cases (see STEP 2
            examples (a)-(g)); null for all other gap types.
  severity: high | medium | low — based on how many cases failed, whether it blocks output entirely,
            and how much the strong/weak structural difference matters.

GAP GROUPING RULE — cases with the same tier1+tier2 label but different
sub_dimensions MUST NOT be merged into the same gap entry.  Create separate
entries — e.g. cases whose root cause is wrong_output vs cases whose root
cause is wrong_strategy vs cases whose root cause is completes_when_weak —
merging them would force a single patch strategy onto distinct root causes.

{_decomp_naming_rules}Do NOT compute scores yourself — submit_gap_report() does that automatically from
failed_case_ids / pass_case_ids.  Systemic determination and recommended_patch_order
are also computed by the tool, so just focus on correctly identifying which cases
exhibit each gap.

If there are previous iterations (see Previous Iteration Evidence in the task prompt),
classify each systemic gap using QUANTITATIVE case counts compared to the most
recent previous iteration's gap report.

  HOW TO MATCH GAPS ACROSS ITERATIONS
  Step A — look for the closest match in iter N-1 (the most recent previous
  gap report).  Matching priority:
    1. Exact match on both category_tier1 AND category_tier2 strings → definite
       match, even if the label wording differs between iterations.
    2. Exact match on category_tier1 + semantic equivalence of category_tier2
       meaning → strong match.
  Step B — if no match exists in iter N-1, search earlier iterations using the
           same priority order.
           If still no match in any previous iteration → NEW.

  COMPARISON BASELINE IS ALWAYS ITER N-1:
  Once a match is found (whether in iter N-1 or an earlier iteration), the
  failed_case_ids count you compare against is ALWAYS taken from iter N-1 —
  NOT from the earlier iteration where the match was first found.
  If the gap was RESOLVED in iter N-1 (failed count = 0), then any non-zero
  failed count in the current iteration is REGRESSION, even if an earlier
  iteration had the same non-zero count.

  HOW TO COMPARE — MANDATORY EXPLICIT CHECK
  For each gap, before assigning status, write out:
    prev_failed = len(failed_case_ids in iter N-1 match)
    curr_failed = len(failed_case_ids in current gap)
    delta = curr_failed - prev_failed
  Then assign:
    delta < 0  → IMPROVEMENT   (count decreased)
    delta = 0  → PERSISTS      (count unchanged)
    delta > 0  → REGRESSION    (count increased)
  If both counts are 0, apply the same logic to pass_case_ids count.
  Do NOT rely on intuition — compute the delta explicitly.

  NEW         — no matching gap found in any previous iteration's report.
                No prior patch to evaluate.

  RESOLVED    — previously had failed_case_ids count > 0, now count = 0 AND
                {resolved_secondary_clause}.  The gap is gone.

  IMPROVEMENT — failed_case_ids count decreased vs the previous iteration
                (or pass_case_ids count decreased when both iters have 0 failed).
                The previous patch moved in the right direction.

  PERSISTS    — failed_case_ids count unchanged vs the previous iteration
                (or pass_case_ids count unchanged when both have 0 failed).
                The previous patch had no measurable effect.

  REGRESSION  — failed_case_ids count increased vs the previous iteration
                (or pass_case_ids count increased when both have 0 failed).
                The previous patch WORSENED this gap.

See STEP 4 — STRATEGY DERIVATION below for the patch strategy per status.

FORBIDDEN for PERSISTS / REGRESSION / IMPROVEMENT / NEW:
Writing skill_patch_hint as "resolved; no patch needed." (or any equivalent
meaning) is ONLY valid for RESOLVED status.  For every other status the
skill_patch_hint MUST describe the concrete patch action.

FORBIDDEN rationalization — "the skill already has text about this":
The presence of an existing instruction in SKILL.md is NOT evidence that the
gap is resolved.  Failed_case_ids count is the ONLY measure.  If count > 0 or
count increased, the existing instruction is insufficient and must be replaced
or escalated — regardless of whether text exists.

{_mechanism_precheck_block}"""




_STEP_4_SUBMIT_GAP_REPORT = """
════════════════════════════════════════════════════════
STEP 4 — SUBMIT GAP REPORT
════════════════════════════════════════════════════════

Submit in TWO steps to avoid token truncation on large reports:

  Step 4a — Write the report JSON to a draft file:
    write_file(
      file_path="{{out_dir}}/gap_report_draft.json",
      content=<the full report JSON string>
    )

  Step 4b — Submit it:
    submit_gap_report(report_file="{{out_dir}}/gap_report_draft.json")

Do NOT include "score" or "recommended_patch_order" in the JSON — these are
computed automatically by the tool from failed_case_ids / pass_case_ids.
The tool writes gap_report.json and returns the computed scores and patch order.

Schema for the report JSON (write this as the content in Step 4a):
{{
  "strong_agent": "<id or 'multiple'>",
  "weak_agent":   "<id or 'multiple'>",
  "outcome_summary": {{"strong": "...", "weak": "..."}},
  "gaps": [
    {{
      "gap_id":                "gap_001",
      "gap_type":              "missing_node | weak_node | extra_wrong_node | order_difference | wrong_skill_routing",
      "sub_dimension":         "short free-text phrase (~3-8 words) describing the weakness — for weak_node only; common examples in STEP 2 are not exhaustive; null for all other gap types",
      "target_skill":          "name of the skill to patch for this gap",
      "category_tier1":        "recover",
      "category_tier2":        "fallback_method",
      "label":                 "missing_recover_fallback",
      "description":           "Strong agent did X ... Weak agent did not ...",
      "weak_evidence":         {{"<case_id>": "<verbatim snippet from that case's weak trajectory>", "...": "..."}},
      "strong_evidence":       {{"<case_id>": "<verbatim snippet from that case's strong trajectory>", "...": "..."}},
      "strong_node_ref":       {{"<case_id>": "node_005", "...": "node_003 or null"}},
      "weak_node_ref":         {{"<case_id>": "node_007 or null", "...": "..."}},
      "strong_completes_when": "...",
      "weak_completes_when":   null,
      "missing_produces":      ["output_file_002"],
      "skill_patch_hint":      "generalizable instruction hint — MUST describe HOW to patch; only write 'resolved; no patch needed.' when status=RESOLVED",
      "severity":              "high | medium | low",
      "failed_case_ids":       ["id1", "id2"],
      "pass_case_ids":         ["id3"],
      "status":                "NEW | IMPROVEMENT | PERSISTS | REGRESSION | RESOLVED"
    }}
  ],
  "summary": "One paragraph ..."
}}

"status" is omitted for iter 0 (no previous iteration to compare against).
For iter ≥ 1, every gap entry must include "status".

"weak_evidence", "strong_evidence", "strong_node_ref", and "weak_node_ref" are dicts keyed by case_id.
EVERY case_id listed in failed_case_ids and pass_case_ids MUST appear as a key in
each of these four dicts.  Use null as the value when a node or snippet is absent
for a particular case.

All gaps — including sub-threshold, single-case, and RESOLVED ones — must appear
in the gaps array so the report is complete.

The tool returns:
  • recommended_patch_order — ordered list of gap_ids to patch (REGRESSION first,
    then PERSISTS, then NEW, then IMPROVEMENT; score descending within each tier)
  • gap_scores — computed score for every gap

MANDATORY after submit_gap_report returns:
{_decomp_step45}
"""
_PHASE_5A_ROOT_CAUSE = """
──────────────────────────────────────────────────────
ROOT CAUSE ANALYSIS & PATCH STRATEGY SELECTION
──────────────────────────────────────────────────────
Using the weak-agent evidence already in the gap report and the structures
surveyed in Steps 2–3, understand what the skill is failing to instruct
correctly and how it causes the failing cases to fail.  Compare weak and
strong agent behavior, as well as the status of the gap and the previous patch 
to determine the correct downstream branch.

**STATUS ANALYSIS** (skip only for NEW gaps):
Answer three questions using the structures pre-loaded at the start of STEP 5
and the exact wording of the previous patch from the previous-iteration
evidence.  Do NOT call read_prev_case_structures again here — use the
observations already in context.  Do NOT re-read trajectory files — use
`code_snippets`, `observation`, and `completes_when` fields already in the
structures.

  1. Why this status? — Explain the MECHANISM, not the arithmetic.
     Quote the exact instruction added/changed for this gap in the previous
     iteration.  Then explain why it failed to change weak-agent behavior,
     using evidence from the current-iteration weak structures:
       • Did the agent's `code_snippets` or `completes_when` show it never
         attempted the step the instruction prescribed?
       • Did the agent attempt the step but in a way that satisfied the letter
         of the instruction without addressing the root cause?
       • Did the instruction target the right behavior in some cases but
         introduce a conflicting constraint in others?
     Cite a specific field from the weak structure (e.g. a `code_snippet` or
     `completes_when` value) that shows what went wrong.

  2. Case-level impact — Using the structures read above, compare the
     previous iteration's weak structure against the current one to explain
     what the previous patch concretely changed (or failed to change) in that
     case's execution:
       • FAIL→PASS: which node or `completes_when` is now present or tighter in
         the current structure that was absent or weaker in the previous one?
         What did the agent now do that it previously skipped?
       • PASS→FAIL (regression): which node changed or appeared in the current
         weak structure that was not there before?  What did the patch cause the
         agent to do differently that broke the case?
       • Stayed FAIL: compare the relevant node (by tier2 label) across both
         structures.  Did the node's `completes_when` or `code_snippets` change
         at all?  If not, the instruction was ignored.  If yes but still failing,
         what does the current structure reveal about why the new behavior is
         still insufficient?
       • Stayed PASS (latent risk): what gap node or weak `completes_when`
         remains in the structure despite the pass verdict?
     Do NOT just list "case_X PASS→FAIL". Every entry must have a reason
     grounded in the diff between the two structures.

  3. Implication — Given the mechanism and case-level evidence above, decide:
       • For REGRESSION: was the patch directionally correct (right idea, wrong
         scope) or wrong direction entirely?  Specify the single replacement
         that works for all cases without over-constraining the regression cases.
       • For PERSISTS: did the previous patch produce different results on
         different cases, or did it have no effect on any case?  If different
         results — which observable condition distinguishes the cases that
         followed it correctly from those that didn't?  If no effect — why
         was the instruction too abstract or too easy to skip?
       • For IMPROVEMENT: what did the previous patch do correctly that must be
         preserved, and what one concrete next step closes the remaining cases?

Record this analysis in the **Status analysis** block of the output protocol.

Before selecting a strategy, check the status assigned in Step 3:

  RESOLVED    → Do NOT patch.  Skip to the next gap in recommended_patch_order.

  REGRESSION  → The previous patch worsened this gap.  Use the STATUS ANALYSIS
                above (case-level delta + previous patch review) to diagnose
                the mechanism, then choose ONE of the following approaches:

                A. DIRECTIONALLY CORRECT, WRONG SCOPE — the previous patch had
                   the right idea but was too broad or too rigid, causing it to
                   break cases it was not designed for.
                   → First check whether the originally-failing cases and the
                     regression cases require DIFFERENT behaviors in the same step:
                     • If YES (different conditions lead to different correct actions):
                       use MULTIPLE CONDITIONAL GATEs (see DRAFTING RULES).
                     • If NO (same behavior needed, just the scope was too broad):
                       write ONE unified replacement that scopes the instruction
                       more precisely.
                   Do NOT stack a new patch on top of the old harmful one —
                   remove or replace the harmful change first.

                B. WRONG DIRECTION ENTIRELY — the previous patch targeted the
                   wrong root cause or introduced a conflicting requirement.
                   → Revert the harmful instruction first.  Then apply a fresh
                     patch using the normal gap-type → strategy mapping for the
                     actual root cause.

  PERSISTS    → GUIDANCE is forbidden.  The previous patch had no measurable
                effect.  Use the STATUS ANALYSIS to diagnose WHY — this drives
                which escalation strategy to pick:

                ⚠ MECHANISM CHECK: The new patch must use a different mechanism
                from the previous one (e.g. if previous was a prose mandate, do
                not write another prose mandate with stronger wording).

                Classify by what the previous patch did to agent behavior across
                cases, then pick the matching escalation:

                • Previous patch produced DIFFERENT results on different cases
                  (some followed it correctly, others ignored it or over-applied
                  it) → a single uniform instruction cannot address all variants.
                  Use MULTIPLE CONDITIONAL GATEs (see DRAFTING RULES).

                • Previous patch was followed uniformly but had no effect on
                  any case → the instruction was present but too abstract:
                  - Agent ignored it entirely → STRUCTURAL RESTRUCTURE
                  - Agent followed letter, not spirit → CODE EXAMPLE or HARD-STOP
                  - Agent did something the skill never explicitly forbids →
                    PROHIBITION
                  - Agent ran the step but skipped sub-steps → MANDATE

                CODE EXAMPLE is REQUIRED when BOTH: (a) implementable action,
                AND (b) strong structures show a consistent generalizable pattern
                across ≥2 cases.  Use descriptive placeholders, no hardcoded values.

                ⚠ DO NOT leave weak text standing: REPLACE it — do not add new
                content alongside it.  Adding new content is acceptable ONLY when
                the patch introduces genuinely new structure (e.g. a new
                conditional branch or checklist gate) that did not exist before.

  IMPROVEMENT → The previous patch direction is correct — preserve it.
                Use the Status Analysis to identify what the remaining failing
                cases are still doing wrong that the newly passing cases are now
                doing correctly:
                  • What specific sub-case or edge condition did the previous
                    patch not cover for the still-failing cases?

                First check whether the remaining failing cases share a single
                root cause that can be addressed by one additional instruction:
                  • If YES (all failing cases fail for the same reason):
                    add a targeted supplement that addresses this shared failure
                    mode WITHOUT changing the part that already fixed the
                    FAIL→PASS cases (a more specific edge case example, an
                    additional condition, or a clarification for a task variant).
                  • If NO (different failing cases fail for different reasons):
                    use MULTIPLE CONDITIONAL GATEs instead (see DRAFTING RULES).

                Do NOT restructure, replace, or add hard-stops or prohibitions
                unless the gap type independently justifies it.  The current
                approach is working — build on it, do not overwrite it.

  NEW         → Apply the normal gap-type → strategy mapping below.

Then determine the patch strategy using the gap type from Steps 2–3 as the
primary signal:

  Gap type → patch strategy mapping:

  missing_node  →  MANDATE
    The weak agent skipped a step entirely.  Add a new mandatory step to the
    Required Workflow with "MUST"/"ALWAYS"/"REQUIRED" language and a completion
    criterion.

  weak_node  →  GUIDANCE, PROHIBITION, PROHIBITION+GUIDANCE, HARD-STOP, MANDATE, or MULTIPLE CONDITIONAL GATEs (choose one):
    First check whether the failing cases under this gap share a single root
    cause that can be addressed by one instruction.  If they fail for DIFFERENT
    reasons (e.g. different task variants trigger different wrong behaviors),
    use MULTIPLE CONDITIONAL GATEs instead of any single-strategy option below.

    MULTIPLE CONDITIONAL GATEs — different cases require different correct behaviors;
                        no single shared instruction can fix all failing cases.
    GUIDANCE          — step ran but used the wrong technique or missed an edge case;
                        also covers (e) wrong_output: patch the code example with
                        the correct computation pattern from the strong trajectory.
    PROHIBITION       — step ran but agent performed an action the skill never forbids.
    PROHIBITION+GUIDANCE — step ran but used the wrong tool (e.g. read_file on a
                        large source document instead of shell(rg/sed)), OR
                        used the right top-level tool in a fragile invocation
                        shape (sub-type (f) shell_invocation_antipattern, e.g.
                        inline `python3 -c "<multi-line>"`): apply PROHIBITION
                        + GUIDANCE together — name the forbidden tool/shape AND
                        provide a concrete alternative (for fragile shell:
                        write_file the script to a `.py` file then `python3
                        path/to/file.py`).
    HARD-STOP         — bad output detected but agent reported success anyway.
    MANDATE           — step ran but was incomplete; critical sub-steps absent.
    → See ★ DRAFTING RULES below for how to write each strategy.

  extra_wrong_node  →  PROHIBITION
    The weak agent performed an extra step absent in the strong agent that
    caused an error (e.g. incorrectly deleted a valid output).  Add a
    "MUST NOT …" or "NEVER …" rule that names the specific erroneous action
    and the condition under which it is forbidden.

  order_difference  →  REORDER
    Fix the sequence of the affected steps in the Required Workflow.

  wrong_skill_routing  →  DESCRIPTION_PATCH
    Extend the correct skill's YAML description (Level 1) with task-type
    keywords derived from the strong agent's skill_selection_reason and the
    weak agent's skill_selection_reason.  Do not rewrite the description —
    preserve its style and extend only.

Write your root cause as THREE parts:
  • Failure mechanism  : why this gap exists in the current skill and how it
      causes the currently failing cases to fail — connect the abstract gap
      to the specific observed failures.
  • Underlying principle : the generalizable behavioral concept that is missing
      or wrong.  For MULTIPLE CONDITIONAL GATEs: also name the observable
      condition that separates the branches.
  • Strategy           : e.g. "GUIDANCE"

A good root cause names a CATEGORY OF SITUATION, not a specific instance.
"""








# ─── Diagnoser-specific NEW prose ───

_BRAINSTORMING_MANDATE = """
════════════════════════════════════════════════════════
MANDATORY FIRST ACTION — ACTIVATE BRAINSTORMING
════════════════════════════════════════════════════════

Before STEP 0 (read_case_history), you MUST call:

  activate_skill(skill_name="brainstorming", reason="<one sentence>")

Then follow the brainstorming skill's 6-phase process to generate 2–3
alternative diagnostic framings of the case pool BEFORE you commit to a
final gap list.  Apply YAGNI: pick the simplest viable framing.

(If you have already activated brainstorming in this session — e.g. on a
resume — do not re-activate; proceed to STEP 0.)
"""

_STEP_4_STRATEGY_HEADER = """
════════════════════════════════════════════════════════
STEP 4 — STRATEGY DERIVATION  (per gap, before submission)
════════════════════════════════════════════════════════

For EACH gap candidate identified in STEP 3, derive the patch strategy.
The result is embedded into the `skill_patch_hint` prose (at least 200 words;
no upper bound) of each gap.  This hint is the binding blueprint that
SkillPatcher (skill_patcher.py) will execute without redoing diagnosis.

Framing: this is "before submitting gap_report" — the strategy decision
is what gets written into each gap's `skill_patch_hint.

"""

_STEP_4_HINT_REQUIREMENTS = """

────────────────────────────────────────────────────────────
SKILL_PATCH_HINT REQUIREMENTS  (the blueprint for SkillPatcher)
────────────────────────────────────────────────────────────

For NEW / IMPROVEMENT / PERSISTS / REGRESSION gaps, each
`skill_patch_hint` MUST be free-form prose (at least 200 words; no upper
bound), covering:

  TARGET    — file(s) + section/header where the patch belongs
  STYLE     — strategy (MANDATE / PROHIBITION / PROHIBITION+GUIDANCE /
              HARD-STOP / GUIDANCE / MULTIPLE CONDITIONAL GATEs /
              REORDER / DESCRIPTION_PATCH) plus any mechanism-switch
              requirement driven by STATUS ANALYSIS
  WORDING   — verbatim instruction wording (descriptive placeholders,
              no hardcoded values).  Use sentences or a bullet list — no
              fixed length cap.  Optionally list 1–2 RELATED FAILURE
              VARIANTS as parallel bullets under the same theme — only
              when they genuinely share the rule's mechanism AND would
              not constrain or harm cases that currently pass.  Do not
              fabricate variants; skip when none clearly apply.  Helps
              the rule generalize to same-mechanism failures unseen in
              training.
              MULTI-MECHANISM — when a gap's failing cases share one
              symptom but span multiple mechanisms (e.g. two cases write
              summaries to a new block, a third creates an extra output
              sheet), WORDING MUST include one patch bullet per
              mechanism under the shared symptom theme.  A single
              generic rule that addresses no mechanism precisely will
              fail again.
  PRIOR-PATCH CRITIQUE — (PERSISTS / REGRESSION / IMPROVEMENT only)
              what the previous patch did and why the new approach differs.
              Use sentences or a bullet list — no fixed length cap.
              Write "n/a" for NEW gaps.

REUSABILITY — the WORDING must be reusable across different tasks, not a
narrow case-specific fix.  Strip case nouns to placeholders (e.g. "the
inspected destination region", not "the Expected sheet at C7:I20").  If the
hint only describes the one failing case, it will neither generalize nor
protect against unseen variants.

BRAINSTORM BEFORE FINALIZING — before locking each hint, you are
encouraged to invoke the brainstorming meta-skill and pick solutions that addresses the
root cause across all this gap's failing cases.

For RESOLVED gaps, set skill_patch_hint to exactly "resolved; no patch
needed."  Only RESOLVED is permitted to use that phrase.
"""

_PATCHER_HANDOFF = """
════════════════════════════════════════════════════════
HANDOFF TO SkillPatcher  (downstream agent)
════════════════════════════════════════════════════════

Your output gap_report.json is consumed by a SEPARATE agent named
SkillPatcher.  SkillPatcher does NOT redo STATUS ANALYSIS or re-select
strategy — it reads each gap's `skill_patch_hint` verbatim as the
binding blueprint for the patch.

Every skill_patch_hint MUST be detailed enough that SkillPatcher can
execute without re-reading case structures.  After STEP 5 SUBMIT, your
session ends.
"""








_POST_SUBMIT_CTA = '  • recommended_patch_order NON-EMPTY → your diagnosis work is complete.\n    The session ends after this step; SkillPatcher will read gap_report.json\n    and apply patches in a separate session.\n  • recommended_patch_order EMPTY → EXIT. No systemic gaps, no patch.'


def _build_diagnoser_system_prompt(
    fail_weight: float = FAIL_WEIGHT,
    pass_weight: float = PASS_WEIGHT,
    score_threshold: float = SCORE_THRESHOLD,
    skills_section: str = "",
    skill_guidance_section: str = "",
) -> str:
    """Build the GapDiagnoser system prompt.

    Composes the diagnosis steps (STEP 0 LOAD HISTORY through STEP 5 SUBMIT)
    with the brainstorming mandate and the SkillPatcher handoff.  Gaps are
    admitted to `recommended_patch_order` by the binary rule; the weighted
    score is still reported per gap, as a diagnostic field only.
    """
    if True:
        filter_rule_describe = (
            f"FILTER RULE (binary mode): a gap is admitted to recommended_patch_order if and "
            f"only if it has ≥{BINARY_MIN_FAILED} FAILING case OR ≥{BINARY_MIN_PASSED} PASSING "
            f"cases.  RESOLVED gaps are excluded.  The 'score' field is still computed for "
            f"diagnostic inspection but is NOT used for filtering — do not reason about the "
            f"score threshold when deciding what to patch."
        )
        resolved_secondary_clause = (
            f"the binary-mode admission test fails (pass_case_ids count < {BINARY_MIN_PASSED}, "
            f"with failed count already 0)"
        )
    # Phase 5A prose has {fail_weight} / {pass_weight} / {score_threshold} placeholders
    phase_5a_body = _PHASE_5A_ROOT_CAUSE.format(
        fail_weight=fail_weight,
        pass_weight=pass_weight,
        score_threshold=score_threshold,
    )
    step_4 = _STEP_4_STRATEGY_HEADER + phase_5a_body + _STEP_4_HINT_REQUIREMENTS

    # The SUBMIT section is authored as STEP 4; it runs last, as STEP 5.
    step_5 = _STEP_4_SUBMIT_GAP_REPORT.replace(
        "STEP 4 — SUBMIT GAP REPORT",
        "STEP 5 — SUBMIT GAP REPORT",
    ).replace(
        "Step 4a — Write the report JSON",
        "Step 5a — Write the report JSON",
    ).replace(
        "Step 4b — Submit it",
        "Step 5b — Submit it",
    ).replace(
        "Schema for the report JSON (write this as the content in Step 4a):",
        "Schema for the report JSON (write this as the content in Step 5a):",
    )

    # MANDATORY after submit_gap_report returns:
    post_submit = _POST_SUBMIT_CTA
    step_5 = step_5.replace(
        "MANDATORY after submit_gap_report returns:\n{_decomp_step45}",
        "MANDATORY after submit_gap_report returns:\n" + post_submit,
    )

    intro = _INTRO_SETUP_GOAL.format(
        fail_weight=fail_weight,
        pass_weight=pass_weight,
        score_threshold=score_threshold,
        filter_rule_describe=filter_rule_describe,
    )

    step_3 = _STEP_3_SYNTHESIS.format(
        score_threshold=score_threshold,
        _decomp_naming_rules="",
        _mechanism_precheck_block="",
        resolved_secondary_clause=resolved_secondary_clause,
    )

    # Rewrite COMPLETION banner for Diagnoser (no patching)
    intro = intro.replace(
        "★ COMPLETION IS MANDATORY — PARTIAL WORK IS NOT ACCEPTABLE ★\n"
        "You MUST patch EVERY gap in recommended_patch_order before ending.\n"
        "Stopping after patching some gaps and leaving others unpatched is a failure.\n"
        "Between each gap patch, your next action MUST be a tool call (not text alone).\n"
        "After patching the final gap, you MUST call finalize_patches() — generating\n"
        "any text-only response before that call will terminate the session prematurely.",
        "★ COMPLETION IS MANDATORY — PARTIAL WORK IS NOT ACCEPTABLE ★\n"
        "You MUST submit a complete gap_report.json covering EVERY systemic gap.\n"
        "Stopping after partial analysis is a failure.  Your final action MUST be\n"
        "submit_gap_report().\n"
        "Generating a text-only response before submit_gap_report will terminate\n"
        "the session prematurely.",
    )

    preface = (
        skills_section + "\n\n"
        + skill_guidance_section + "\n\n"
        + _BRAINSTORMING_MANDATE
    )

    parts = [
        preface,
        intro,
        _SKILL_ARCHITECTURE_BANNER,
        _STEP_0_LOAD_HISTORY,
        _STEP_1_SURVEY.format(fail_weight=fail_weight, pass_weight=pass_weight),
        _STEP_2_DEEP_DIVE,
        step_3,
        step_4,
        step_5,
        _PATCHER_HANDOFF,
    ]
    return "\n".join(p.rstrip("\n") for p in parts if p) + "\n"


# ─── Diagnoser class ───

class GapDiagnoser(GapAgentBase):
    _AGENT_NAME = "GapDiagnoser"
    _TEXT_ONLY_PRE_REPORT_RESUME_INPUT = (
        "You stopped early without submitting the gap report. "
        "You have NOT called submit_gap_report yet — gap_report.json does not exist. "
        "Write your gap analysis to gap_report_draft.json and call submit_gap_report now. "
        "DO NOT produce another text-only response — use tool calls."
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
        min_gap_count: int = 1,
        meta_skills_dir: str | Path | None = "skills/meta_brainstorming",
    ):
        # Gap admission is binary (see `_build_diagnoser_system_prompt`); the
        # weights below only scale the per-gap diagnostic score in the report.
        fail_weight = FAIL_WEIGHT
        pass_weight = PASS_WEIGHT
        score_threshold = SCORE_THRESHOLD

        self._meta_mgr = SkillManager()
        try:
            self._meta_mgr._skills = load_skills_from_dir(str(meta_skills_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the brainstorming meta-skill from {meta_skills_dir}: {exc}"
            ) from exc
        if not self._meta_mgr.get_skills():
            raise RuntimeError(
                f"No meta-skill found in {meta_skills_dir}.  The Diagnoser activates the "
                f"'brainstorming' meta-skill, which ships in skills/meta_brainstorming/.  "
                f"Pass --meta-skills-dir if yours lives elsewhere."
            )
        skills_xml = render_agent_skills(self._meta_mgr.get_skills())
        guidance_xml = mandate_skill_guidance(
            has_skills=bool(self._meta_mgr.get_skills()),
            require_skill=True,
            multi_skill=False,
        )

        if system_prompt_multi is None:
            system_prompt_multi = _build_diagnoser_system_prompt(
                fail_weight=fail_weight,
                pass_weight=pass_weight,
                score_threshold=score_threshold,
                skills_section=skills_xml,
                skill_guidance_section=guidance_xml,
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

    def _build_tools(self) -> list:
        tools = [self._make_read_file_tool(), self._make_write_file_tool()]
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
        all_tools = super()._build_multi_tools(
            cases=cases,
            case_history_path=case_history_path,
            out_dir=out_dir,
            prev_gap_narrative_paths=prev_gap_narrative_paths,
        )
        return [t for t in all_tools if getattr(t, "name", "") != "finalize_patches"]

    def _build_prompt_multi(
        self,
        task_description: str,
        cases: list[CaseContext],
        skill_paths: list[str],
        out_dir: str,
        force_read_trajectories: bool = False,
        prev_gap_trajectory_paths: list[str] | None = None,
        prev_gap_patch_narrative_paths: list[str] | None = None,
        **_: Any,
    ) -> str:
        n = len(cases)
        skill_listing = _expand_skill_listing(skill_paths)
        traj_policy = (
            "TRAJECTORY POLICY: For EVERY case, call read_case_trajectory(case_id, 'strong') "
            "AND read_case_trajectory(case_id, 'weak') before Step 2 analysis."
            if force_read_trajectories
            else "TRAJECTORY POLICY: Call read_case_trajectory only when the code_snippets "
                 "in the execution structures are insufficient to understand a specific gap."
        )
        prev_section = ""
        narrative_list = prev_gap_patch_narrative_paths or []
        traj_list = prev_gap_trajectory_paths or []
        if narrative_list:
            lines = ["\n## Previous Iteration Evidence\n"]
            for i, p in enumerate(narrative_list):
                report_path = str(Path(p).parent / "gap_report.json")
                lines.append(f"  Iteration {i+1} — gap report:      {report_path}")
                lines.append(f"  Iteration {i+1} — patch narrative: {p}")
            prev_section = "\n".join(lines) + "\n"
        elif traj_list:
            lines = ["\n## Previous Iteration Evidence\n"]
            for i, p in enumerate(traj_list):
                fmt = "json" if p.endswith(".json") else "md"
                lines.append(f"  Iteration {i+1} — trajectory ({fmt}): {p}")
            prev_section = "\n".join(lines) + "\n"

        return f"""Task Description:
{task_description}

You have {n} training cases where the strong agent succeeded but the weak agent failed (or are at-risk).

Skills available — SkillPatcher will modify these based on your gap_report.json:
{skill_listing}

Gap report output directory: {out_dir}
  Draft file for STEP 5 submission: {out_dir}/gap_report_draft.json

{traj_policy}
{prev_section}
Complete STEPs 0–5 using TOOLS, not prose.
You do NOT patch any files — SkillPatcher reads gap_report.json afterwards.

Begin by activating the brainstorming meta-skill (if available), then call
read_case_history() to start STEP 0.
"""

    def run_streamed_diagnoser(self, **kwargs: Any):
        return self.run_streamed_multi(**kwargs)


__all__ = [
    "GapDiagnoser",
    "_build_diagnoser_system_prompt",
]
