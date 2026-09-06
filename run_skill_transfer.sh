#!/usr/bin/env bash
#
# Skill Transfer — adapt a skill that already works for a strong agent so that a
# specific weak agent can execute it, without touching the weak agent's weights.
#
# Per iteration:
#   1. run the weak agent on the training tasks under the current skill
#   2. abstract both agents' trajectories into typed execution structures
#   3. GapDiagnoser  localises the gaps and writes a ranked gap report
#   4. SkillPatcher  applies every patch hint to the skill in one pass
#   5. score the new skill on the validation split and keep the best iterate
#
# The strong agent runs once, before the loop; only tasks it solves are kept.
#
# Usage:
#   export OPENAI_API_KEY=...            # or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
#   bash run_skill_transfer.sh \
#       --dataset-dir  data/spreadsheetbench/train_20 \
#       --val-dir      data/spreadsheetbench/validation_30 \
#       --out-dir      runs/sb_gpt41mini \
#       --skills-dir   skills/xlsx_baseline \
#       --skill-names  xlsx \
#       --strong-model gpt-5.4 \
#       --weak-model   gpt-4.1-mini \
#       --max-edits    6
#
# Run `bash run_skill_transfer.sh --help` for the full flag list.

# Activate the conda env (Python 3.11 + LibreOffice). See setup_env.sh for overrides.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env.sh"
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.11}"
HELPERS="${SCRIPT_DIR}/skill_transfer/pipeline_helpers.py"

DATASET_DIR=""
OUT_DIR=""
# SKILLS_DIR is the read-only BASELINE source from which per-run working_skills/
# gets bootstrapped at Step 1.  Pipeline never modifies SKILLS_DIR — all agent
# edits land in WORKING_SKILLS_DIR (derived later as ${OUT_DIR}/working_skills/)
# so concurrent / sequential runs can't pollute each other through a shared
# live skill dir.  Default points at the canonical baseline.
SKILLS_DIR="${SCRIPT_DIR}/skills/xlsx_baseline"
WORKING_SKILLS_DIR=""        # derived after OUT_DIR is set
RESET_WORKING_SKILLS=""      # set via --reset-working-skills to force re-copy from SKILLS_DIR

STRONG_MODEL="gpt-5.4"
WEAK_MODEL="gpt-5.4-mini"
GAP_MODEL="gpt-5.4"
ABSTRACTOR_MODEL="gpt-5.4"
VALIDATOR_MODEL="gpt-5.4-mini"

STRONG_REASONING_EFFORT=""      # reasoning effort for strong agent (low/medium/high, empty=off)
STRONG_MAX_ATTEMPTS=5            # max times strong agent retries failing cases (1 = single attempt; 3 = retry twice on still-failing subset)
WEAK_REASONING_EFFORT=""        # reasoning effort for weak agent
GAP_REASONING_EFFORT=""         # reasoning effort for gap agent
GAP_MAX_OUTPUT_TOKENS=1280000     # max output tokens per gap agent response (0 = model default)
GAP_AGENT_TIMEOUT=1000            # subprocess timeout in seconds for the gap agent (0 = no limit)

DIAGNOSER_MODEL=""               # empty = inherit --gap-model
PATCHER_MODEL=""                 # empty = inherit --gap-model
DIAGNOSER_TIMEOUT=""             # empty = inherit --gap-agent-timeout
PATCHER_TIMEOUT=""               # empty = inherit --gap-agent-timeout
PATCHER_MODE="one-shot"           # per-gap (one edit per gap) | one-shot (all gaps in one batched patch)
ABSTRACTOR_REASONING_EFFORT=""  # reasoning effort for trajectory abstractor
VALIDATOR_REASONING_EFFORT=""   # reasoning effort for validator

BATCH_SIZE=""          # empty = all tasks
MAX_EDITS=3
MAX_WORKERS=4
MAX_TURNS=20
MAX_INPUT_TOKENS_PER_TURN=1000000000

MIN_GAP_COUNT=1        # minimum failed+pass case count for a gap to enter patch order (split mode only)

SIMPLE_ONLY=""         # "--simple-only" flag for prepare
SPLIT=""               # "--split <name>" flag for prepare (e.g. train_20)
SPLIT_MANIFEST=""      # "--split-manifest <path>" flag for prepare (optional)
SEED=42                # "--seed <int>" flag for prepare (random shuffle; omit for dataset order)
SKIP_STRONG=""         # set to "1" to skip strong agent run
TRAJECTORY_FORMAT="json" # "md" or "json" — which trajectory format the gap agent reads

VAL_DIR=""             # validation dataset dir (optional); enables per-iter val eval
VAL_MAX_CASES=""       # max validation cases to evaluate (empty = all)
VAL_OUT_DIR=""         # set automatically to ${OUT_DIR}/val_evals
VAL_CASE_IDS=()        # populated from VAL_DIR/dataset.json
_val_n=0               # monotonic counter for val eval iteration dirs
RESUME=""              # set to "1" to skip any step whose outputs already exist
REQUIRE_SKILL=""       # "--require-skill" flag forwarded to run-agents
MULTI_SKILL=""         # "--multi-skill"   flag forwarded to run-agents

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)        DATASET_DIR="$2";       shift 2 ;;
        --out-dir)            OUT_DIR="$2";            shift 2 ;;
        --skills-dir)         SKILLS_DIR="$2";         shift 2 ;;
        --baseline-skills-dir) SKILLS_DIR="$2";        shift 2 ;;   # alias, more explicit name
        --reset-working-skills) RESET_WORKING_SKILLS="1"; shift 1 ;;  # force re-copy from SKILLS_DIR
        --strong-model)       STRONG_MODEL="$2";       shift 2 ;;
        --weak-model)         WEAK_MODEL="$2";         shift 2 ;;
        --gap-model)          GAP_MODEL="$2";          shift 2 ;;
        --abstractor-model)   ABSTRACTOR_MODEL="$2";   shift 2 ;;
        --validator-model)    VALIDATOR_MODEL="$2";    shift 2 ;;
        --strong-reasoning-effort)     STRONG_REASONING_EFFORT="$2";     shift 2 ;;
        --strong-max-attempts)         STRONG_MAX_ATTEMPTS="$2";         shift 2 ;;
        --weak-reasoning-effort)       WEAK_REASONING_EFFORT="$2";       shift 2 ;;
        --gap-reasoning-effort)        GAP_REASONING_EFFORT="$2";        shift 2 ;;
        --gap-max-output-tokens)       GAP_MAX_OUTPUT_TOKENS="$2";       shift 2 ;;
        --gap-agent-timeout)           GAP_AGENT_TIMEOUT="$2";           shift 2 ;;
        --diagnoser-model)             DIAGNOSER_MODEL="$2";             shift 2 ;;
        --patcher-model)               PATCHER_MODEL="$2";               shift 2 ;;
        --diagnoser-timeout)           DIAGNOSER_TIMEOUT="$2";           shift 2 ;;
        --patcher-timeout)             PATCHER_TIMEOUT="$2";             shift 2 ;;
        --abstractor-reasoning-effort) ABSTRACTOR_REASONING_EFFORT="$2"; shift 2 ;;
        --validator-reasoning-effort)  VALIDATOR_REASONING_EFFORT="$2";  shift 2 ;;
        --batch-size)         BATCH_SIZE="$2";         shift 2 ;;
        --max-edits)          MAX_EDITS="$2";          shift 2 ;;
        --max-workers)        MAX_WORKERS="$2";        shift 2 ;;
        --max-turns)                  MAX_TURNS="$2";                    shift 2 ;;
        --max-input-tokens-per-turn)  MAX_INPUT_TOKENS_PER_TURN="$2";   shift 2 ;;
        --min-gap-count)              MIN_GAP_COUNT="$2";                shift 2 ;;
        --simple-only)        SIMPLE_ONLY="--simple-only"; shift 1 ;;
        --split)              SPLIT="--split $2";            shift 2 ;;
        --split-manifest)     SPLIT_MANIFEST="--split-manifest $2"; shift 2 ;;
        --seed)               SEED="$2";                     shift 2 ;;
        --skip-strong)        SKIP_STRONG="1";         shift 1 ;;
        --trajectory-format)  TRAJECTORY_FORMAT="$2";  shift 2 ;;
        --resume)             RESUME="1";              shift 1 ;;
        --val-dir)            VAL_DIR="$2";            shift 2 ;;
        --val-max-cases)      VAL_MAX_CASES="$2";      shift 2 ;;
        --require-skill)      REQUIRE_SKILL="--require-skill"; shift 1 ;;
        --multi-skill)        MULTI_SKILL="--multi-skill";     shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
[[ -z "$DATASET_DIR" ]] && { echo "ERROR: --dataset-dir is required" >&2; exit 1; }
[[ -z "$OUT_DIR" ]]     && { echo "ERROR: --out-dir is required"     >&2; exit 1; }
[[ -f "$DATASET_DIR/dataset.json" ]] || { echo "ERROR: dataset.json not found in $DATASET_DIR" >&2; exit 1; }
[[ -d "$SKILLS_DIR" ]] || { echo "ERROR: --skills-dir / --baseline-skills-dir path not found: $SKILLS_DIR" >&2; exit 1; }

# ── Derive WORKING_SKILLS_DIR from OUT_DIR ────────────────────────────────────
# All agent invocations read+write WORKING_SKILLS_DIR; SKILLS_DIR is just the
# read-only baseline source.  This isolates each run's edits from other runs.
WORKING_SKILLS_DIR="${OUT_DIR}/working_skills"

# Inherit per-agent model / timeout from --gap-model / --gap-agent-timeout when not overridden
[[ -z "$DIAGNOSER_MODEL"   ]] && DIAGNOSER_MODEL="$GAP_MODEL"
[[ -z "$PATCHER_MODEL"     ]] && PATCHER_MODEL="$GAP_MODEL"
[[ -z "$DIAGNOSER_TIMEOUT" ]] && DIAGNOSER_TIMEOUT="$GAP_AGENT_TIMEOUT"
[[ -z "$PATCHER_TIMEOUT"   ]] && PATCHER_TIMEOUT="$GAP_AGENT_TIMEOUT"

mkdir -p "$OUT_DIR"

# ── Reasoning effort arg strings (empty when effort not set) ──────────────────
STRONG_REASON_ARG=""; [[ -n "$STRONG_REASONING_EFFORT" ]] && STRONG_REASON_ARG="--reasoning-effort $STRONG_REASONING_EFFORT"
WEAK_REASON_ARG="";   [[ -n "$WEAK_REASONING_EFFORT" ]]   && WEAK_REASON_ARG="--reasoning-effort $WEAK_REASONING_EFFORT"
GAP_REASON_ARG="";    [[ -n "$GAP_REASONING_EFFORT" ]]    && GAP_REASON_ARG="--reasoning-effort $GAP_REASONING_EFFORT"
GAP_MAX_TOK_ARG="";  [[ "${GAP_MAX_OUTPUT_TOKENS:-0}" -gt 0 ]] && GAP_MAX_TOK_ARG="--max-output-tokens $GAP_MAX_OUTPUT_TOKENS"
# Cross-platform timeout wrapper for the gap agent subprocess
# Generic timeout wrapper used by the split-framework Diagnoser / Patcher
# subprocesses.  Arg 1 is the timeout in seconds (0 = no limit); remaining
# args are the command to run.  Same cross-platform fallback as above.
_run_with_timeout() {
    local _t="$1"; shift
    if [[ "${_t:-0}" -gt 0 ]]; then
        if command -v timeout &>/dev/null; then
            timeout "$_t" "$@"
        elif command -v gtimeout &>/dev/null; then
            gtimeout "$_t" "$@"
        else
            "$@"
        fi
    else
        "$@"
    fi
}
ABST_REASON_ARG="";   [[ -n "$ABSTRACTOR_REASONING_EFFORT" ]] && ABST_REASON_ARG="--reasoning-effort $ABSTRACTOR_REASONING_EFFORT"
VAL_REASON_ARG="";    [[ -n "$VALIDATOR_REASONING_EFFORT" ]]  && VAL_REASON_ARG="--reasoning-effort $VALIDATOR_REASONING_EFFORT"

# ── Tee all output to a timestamped log file ──────────────────────────────────
_LOG_FILE="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$_LOG_FILE") 2>&1
echo "Logging to: $_LOG_FILE"

# ── API backend banner ────────────────────────────────────────────────────────
$PYTHON - <<'PYEOF'
import os, sys
BOLD="\033[1m"; RESET="\033[0m"; CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; DIM="\033[90m"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
if endpoint:
    key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        print(f"{YELLOW}WARNING:{RESET} Azure mode active but no API key found (AZURE_OPENAI_API_KEY / OPENAI_API_KEY).", file=sys.stderr)
        sys.exit(1)
    key_hint = key[:8] + "..." + key[-4:]
    print(f"  {DIM}API backend :{RESET} {BOLD}{CYAN}Azure OpenAI{RESET}  endpoint={CYAN}{endpoint}{RESET}  key={DIM}{key_hint}{RESET}")
else:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print(f"{YELLOW}WARNING:{RESET} OPENAI_API_KEY is not set — API calls will fail.", file=sys.stderr)
        sys.exit(1)
    key_hint = key[:8] + "..." + key[-4:]
    print(f"  {DIM}API backend :{RESET} {BOLD}{GREEN}Standard OpenAI{RESET}  key={DIM}{key_hint}{RESET}")
PYEOF

# ── Colour helpers ────────────────────────────────────────────────────────────
BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RED="\033[31m"; DIM="\033[90m"
banner() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"; echo -e "  ${BOLD}$1${RESET}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"; }
step()   { echo -e "\n${BOLD}${CYAN}▶${RESET} ${BOLD}$1${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()   { echo -e "  ${YELLOW}!${RESET} $1"; }
fail()   { echo -e "  ${RED}✗${RESET} $1"; }

# ── Helper: all-pass check ─────────────────────────────────────────────────────
# Returns exit code 0 if all case_ids in eval_file have "pass": true
all_pass() {
    local eval_file="$1"; shift
    local case_ids=("$@")
    $PYTHON - "$eval_file" "${case_ids[@]}" <<'PYEOF'
import json, sys
eval_file = sys.argv[1]
case_ids  = sys.argv[2:]
data = json.load(open(eval_file))
ok = all(data.get(cid, {}).get("pass", False) for cid in case_ids)
sys.exit(0 if ok else 1)
PYEOF
}

# ── Helper: count pass/fail in eval file ──────────────────────────────────────
count_pass() {
    local eval_file="$1"; shift
    local case_ids=("$@")
    $PYTHON - "$eval_file" "${case_ids[@]}" <<'PYEOF'
import json, sys
eval_file = sys.argv[1]
case_ids  = sys.argv[2:]
data = json.load(open(eval_file))
passed = sum(1 for cid in case_ids if data.get(cid, {}).get("pass", False))
print(f"{passed}/{len(case_ids)}")
PYEOF
}

# ── Step cost printer ────────────────────────────────────────────────────────
# _print_step_cost <label> <glob_pattern>...
#   Globs for cost JSON files matching the given pattern(s), sums total_cost_usd
#   and total tokens, and prints a one-line summary.
_print_step_cost() {
    local label="$1"; shift
    local patterns=("$@")
    $PYTHON - "$label" "${patterns[@]}" <<'PYEOF'
import json, sys, glob, os

label    = sys.argv[1]
patterns = sys.argv[2:]

RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"; DIM = "\033[90m"; CYAN = "\033[36m"

cost_files = []
for pat in patterns:
    cost_files.extend(glob.glob(pat, recursive=True))
cost_files = sorted(set(cost_files))

if not cost_files:
    print(f"  {DIM}[cost] {label}: no cost files found{RESET}")
    sys.exit(0)

total_in, total_out, total_cost = 0, 0, 0.0
for cf in cost_files:
    try:
        data = json.load(open(cf))
        # Support both CostSummary format (has "runs") and single CostRun format
        runs = data.get("runs", [data] if "total_cost_usd" in data else [])
        for r in runs:
            total_in   += r.get("input_tokens",  0)
            total_out  += r.get("output_tokens", 0)
            total_cost += r.get("total_cost_usd", 0.0)
    except Exception:
        pass

print(f"  {DIM}[cost] {label}: {total_in:,} in + {total_out:,} out = {BOLD}{GREEN}${total_cost:.4f}{RESET}")
PYEOF
}

# ── Validation eval helper ───────────────────────────────────────────────────
# _do_val_eval <label> <iter_n>
#   label  : human label written to filenames, e.g. "iter_00" or "final"
#   iter_n : numeric iteration passed to run-agents / eval (0, 1, 2, …)
# Runs the weak agent on VAL_CASE_IDS with the current SKILLS_DIR, evals pass
# rate, snapshots all detected skill dirs, and updates val_eval_summary.json.
_do_val_eval() {
    local label="$1"
    local iter_n="$2"
    local eval_file="${VAL_OUT_DIR}/weak_val_${label}.json"
    local snap_dir="${VAL_OUT_DIR}/skill_snap_${label}"
    local iter_str; iter_str=$(printf '%02d' "$iter_n")
    local val_cases_root="${VAL_OUT_DIR}/cases"

    # Snapshot entire skill directories (all files, not just SKILL.md).
    # On --resume, skip if the snapshot already exists — overwriting it would replace
    # the original checkpoint-time skill with the current (post-all-patches) live skill,
    # corrupting the best-checkpoint selection.
    if [[ -d "$snap_dir" ]] && [[ -n "$RESUME" ]]; then
        warn "Skipping skill snapshot ${label} (--resume, snapshot exists)"
    else
        rm -rf "$snap_dir"
        mkdir -p "$snap_dir"
        for skill in "${ALL_SKILL_NAMES[@]}"; do
            cp -r "${WORKING_SKILLS_DIR}/${skill}" "${snap_dir}/"
        done
    fi

    if [[ -n "$RESUME" && -f "$eval_file" ]]; then
        warn "Skipping val eval ${label} (--resume, file exists)"
    else
        # In resume mode, only run val cases that are missing a trajectory
        local VAL_RUN_IDS=("${VAL_CASE_IDS[@]}")
        if [[ -n "$RESUME" ]]; then
            VAL_RUN_IDS=()
            for _cid in "${VAL_CASE_IDS[@]}"; do
                [[ -f "${val_cases_root}/${_cid}/iter_${iter_str}/weak_trajectory.md" ]] || VAL_RUN_IDS+=("$_cid")
            done
        fi

        if [[ ${#VAL_RUN_IDS[@]} -eq 0 ]]; then
            warn "Skipping val agent ${label} (all trajectories exist)"
        else
            [[ ${#VAL_RUN_IDS[@]} -lt ${#VAL_CASE_IDS[@]} ]] && \
                warn "Resuming val agent ${label}: ${#VAL_RUN_IDS[@]} / ${#VAL_CASE_IDS[@]} cases missing"
            step "Validation eval (${label}, ${#VAL_RUN_IDS[@]} / ${#VAL_CASE_IDS[@]} cases)…"
            $PYTHON "$HELPERS" run-agents \
                --out-dir     "$VAL_OUT_DIR" \
                --model       "$WEAK_MODEL" \
                --role        weak \
                --iter        "$iter_n" \
                --skills-dir  "$WORKING_SKILLS_DIR" \
                --max-turns   "$MAX_TURNS" \
                --max-input-tokens-per-turn "$MAX_INPUT_TOKENS_PER_TURN" \
                --max-workers "$MAX_WORKERS" \
                --case-ids    "${VAL_RUN_IDS[@]}" \
                $REQUIRE_SKILL $MULTI_SKILL $WEAK_REASON_ARG
        fi

        $PYTHON "$HELPERS" eval \
            --dataset-dir "$VAL_DIR" \
            --out-dir     "$VAL_OUT_DIR" \
            --iter        "$iter_n" \
            --eval-out    "$eval_file" \
            --case-ids    "${VAL_CASE_IDS[@]}"
    fi

    local pass_count
    pass_count="$(count_pass "$eval_file" "${VAL_CASE_IDS[@]}")"
    ok "Val eval (${label}): ${pass_count} / ${#VAL_CASE_IDS[@]} passed"
    _print_step_cost "val-eval ${label}" "${VAL_OUT_DIR}/cases/**/iter_$(printf '%02d' "$iter_n")/cost_weak.json"

    # Append result to val_eval_summary.json
    $PYTHON - "$label" "$pass_count" "${#VAL_CASE_IDS[@]}" "${VAL_OUT_DIR}/val_eval_summary.json" <<'PYEOF'
import json, sys, os
label     = sys.argv[1]
pass_str  = sys.argv[2]   # "X/Y"
total     = int(sys.argv[3])
out_file  = sys.argv[4]
passed    = int(pass_str.split("/")[0])
data      = {}
if os.path.exists(out_file):
    try:    data = json.load(open(out_file))
    except: pass
data[label] = {"passed": passed, "total": total,
               "pass_rate": round(passed / total, 4) if total else 0.0}
json.dump(data, open(out_file, "w"), indent=2)
PYEOF
}

# ── Resume helpers ───────────────────────────────────────────────────────────
# These reference BATCH_CASES_ROOT which is set per-batch inside the batch loop.
# _all_trajs_exist <iter> <case_id>...  → 0 if every weak_trajectory.md exists
_all_trajs_exist() {
    local iter_str; iter_str=$(printf '%02d' "$1"); shift
    for cid in "$@"; do
        [[ -f "${BATCH_CASES_ROOT}/${cid}/iter_${iter_str}/weak_trajectory.md" ]] || return 1
    done
    return 0
}
# _all_reports_exist <iter> <case_id>...  → 0 if every validation_report.json exists
_all_reports_exist() {
    local iter_str; iter_str=$(printf '%02d' "$1"); shift
    for cid in "$@"; do
        [[ -f "${BATCH_CASES_ROOT}/${cid}/iter_${iter_str}/validator/validation_report.json" ]] || return 1
    done
    return 0
}
# _all_structures_exist <iter> <case_id>...  → 0 if every per-case weak_structure.json exists
_all_structures_exist() {
    local iter_str; iter_str=$(printf '%02d' "$1"); shift
    for cid in "$@"; do
        [[ -f "${BATCH_CASES_ROOT}/${cid}/iter_${iter_str}/weak_structure.json" ]] || return 1
    done
    return 0
}
# _missing_cases <iter> <rel_path> <case_id>...
# Prints (one per line) the case IDs whose file batch_NN/cases/{id}/iter_NN/<rel_path> is absent.
_missing_cases() {
    local iter_str; iter_str=$(printf '%02d' "$1"); shift
    local rel="$1"; shift
    for cid in "$@"; do
        [[ -f "${BATCH_CASES_ROOT}/${cid}/iter_${iter_str}/${rel}" ]] || echo "$cid"
    done
}

# _dir_hash <dir>
# Compute a single hash over all non-.DS_Store files in a skill directory.
# Used to detect whether any file in the skill dir changed after the gap agent.
_dir_hash() {
    find "$1" -type f -not -name ".DS_Store" | sort | while IFS= read -r f; do cat "$f"; done \
        | (md5 -q 2>/dev/null || md5sum | awk '{print $1}')
}

# _snapshot_skill <src_dir> <dest_dir>
# Copy an entire skill directory to dest_dir, removing any prior snapshot first.
_snapshot_skill() {
    rm -rf "$2"
    cp -r "$1" "$2"
}

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Prepare case directories
# ══════════════════════════════════════════════════════════════════════════════
banner "Step 1 — Prepare Cases"

CASES_ROOT="${OUT_DIR}/cases"          # global: inputs + strong data (never split)
GLOBAL_CASES_ROOT="${OUT_DIR}/cases"   # alias; passed to gap agent as --global-cases-dir
CASE_IDS_FILE="${OUT_DIR}/case_ids.txt"

if [[ -n "$RESUME" && -s "$CASE_IDS_FILE" ]]; then
    warn "Skipping prepare (--resume, ${CASE_IDS_FILE} exists)"
else
    step "Preparing ALL case directories from dataset…"
    $PYTHON "$HELPERS" prepare \
        --dataset-dir "$DATASET_DIR" \
        --out-dir     "$OUT_DIR" \
        $SIMPLE_ONLY \
        $SPLIT \
        $SPLIT_MANIFEST \
        ${SEED:+--seed "$SEED"} \
        > "$CASE_IDS_FILE"
fi

CASE_IDS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] && CASE_IDS+=("$line")
done < "$CASE_IDS_FILE"
ok "Prepared ${#CASE_IDS[@]} cases → ${CASES_ROOT}"

[[ ${#CASE_IDS[@]} -eq 0 ]] && { fail "No cases prepared — check dataset.json"; exit 1; }

# ── Prepare validation cases (if --val-dir provided) ─────────────────────────
if [[ -n "$VAL_DIR" ]]; then
    [[ -f "$VAL_DIR/dataset.json" ]] || { fail "--val-dir: dataset.json not found in $VAL_DIR"; exit 1; }
    VAL_OUT_DIR="${OUT_DIR}/val_evals"
    VAL_CASES_FILE="${OUT_DIR}/val_case_ids.txt"
    mkdir -p "$VAL_OUT_DIR"

    if [[ -n "$RESUME" && -s "$VAL_CASES_FILE" ]]; then
        warn "Skipping val prepare (--resume, ${VAL_CASES_FILE} exists)"
    else
        step "Preparing validation cases from ${VAL_DIR}…"
        $PYTHON "$HELPERS" prepare \
            --dataset-dir "$VAL_DIR" \
            --out-dir     "$VAL_OUT_DIR" \
            > "$VAL_CASES_FILE"
    fi

    VAL_ALL_IDS=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -n "$line" ]] && VAL_ALL_IDS+=("$line")
    done < "$VAL_CASES_FILE"

    if [[ -n "$VAL_MAX_CASES" ]] && [[ "${#VAL_ALL_IDS[@]}" -gt "$VAL_MAX_CASES" ]]; then
        VAL_CASE_IDS=("${VAL_ALL_IDS[@]:0:$VAL_MAX_CASES}")
    else
        VAL_CASE_IDS=("${VAL_ALL_IDS[@]}")
    fi
    ok "${#VAL_CASE_IDS[@]} validation cases ready → ${VAL_OUT_DIR}"
fi

# ── Bootstrap WORKING_SKILLS_DIR from baseline SKILLS_DIR ────────────────────
# Per-run isolation: every agent reads/writes WORKING_SKILLS_DIR; SKILLS_DIR
# stays read-only.  On --resume, if WORKING_SKILLS_DIR already exists we keep
# the in-progress state (continuing from where the previous attempt stopped).
if [[ ! -d "$WORKING_SKILLS_DIR" || -n "$RESET_WORKING_SKILLS" ]]; then
    if [[ -d "$WORKING_SKILLS_DIR" ]]; then
        warn "--reset-working-skills: removing existing $WORKING_SKILLS_DIR"
        rm -rf "$WORKING_SKILLS_DIR"
    fi
    step "Bootstrapping working_skills from baseline (${SKILLS_DIR})…"
    mkdir -p "$(dirname "$WORKING_SKILLS_DIR")"
    cp -R "$SKILLS_DIR" "$WORKING_SKILLS_DIR"
    # Manifest for audit
    _baseline_md5_file=""
    for _s in "$SKILLS_DIR"/*; do
        _md=$(find "$_s" -name SKILL.md -maxdepth 2 2>/dev/null | head -1)
        [[ -n "$_md" ]] && _baseline_md5_file="${_baseline_md5_file}$(md5sum "$_md" | awk '{print $1}') $(basename "$_s")\n"
    done
    cat > "$WORKING_SKILLS_DIR/.baseline_manifest.json" <<EOF
{
  "baseline_source": "$SKILLS_DIR",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "baseline_md5s": "$(printf "$_baseline_md5_file" | tr '\n' ';' | sed 's/;$//')",
  "out_dir": "$OUT_DIR"
}
EOF
    ok "working_skills initialized → $WORKING_SKILLS_DIR"
else
    ok "Reusing existing working_skills → $WORKING_SKILLS_DIR (use --reset-working-skills to re-copy from baseline)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Run strong agent
# ══════════════════════════════════════════════════════════════════════════════
banner "Step 2 — Strong Agent (${STRONG_MODEL})"

STRONG_EVAL="${OUT_DIR}/strong_eval.json"

if [[ ( -n "$SKIP_STRONG" || -n "$RESUME" ) && -f "$STRONG_EVAL" ]]; then
    warn "Skipping strong agent run ($STRONG_EVAL exists)"
else
    # ── Retry loop: up to STRONG_MAX_ATTEMPTS rounds; each round runs the
    # strong agent only on cases that did NOT pass the previous round.
    # Archives previous-attempt artifacts into cases/<id>/strong_attempt_NN/
    # so the main cases/<id>/strong_* always reflects the most recent (and
    # ideally passing) attempt — used by downstream pipeline stages.
    _STRONG_REMAINING=("${CASE_IDS[@]}")
    _STRONG_ATTEMPT=1
    while [[ "$_STRONG_ATTEMPT" -le "$STRONG_MAX_ATTEMPTS" && ${#_STRONG_REMAINING[@]} -gt 0 ]]; do
        if [[ "$_STRONG_ATTEMPT" -gt 1 ]]; then
            step "Strong attempt ${_STRONG_ATTEMPT}/${STRONG_MAX_ATTEMPTS} — archiving previous artifacts for ${#_STRONG_REMAINING[@]} still-failing cases…"
            _PREV_ATTEMPT=$(printf "%02d" $(( _STRONG_ATTEMPT - 1 )))
            for _cid in "${_STRONG_REMAINING[@]}"; do
                _cdir="${CASES_ROOT}/${_cid}"
                _adir="${_cdir}/strong_attempt_${_PREV_ATTEMPT}"
                mkdir -p "$_adir"
                for _art in strong_trajectory.md strong_trajectory.json strong_structure.json strong_skills_used.json cost_strong.json; do
                    [[ -f "${_cdir}/${_art}" ]] && mv "${_cdir}/${_art}" "${_adir}/${_art}"
                done
                [[ -d "${_cdir}/strong_outputs" ]] && mv "${_cdir}/strong_outputs" "${_adir}/strong_outputs"
            done
        fi

        step "Running strong agent (attempt ${_STRONG_ATTEMPT}/${STRONG_MAX_ATTEMPTS}) on ${#_STRONG_REMAINING[@]} case(s) (max_workers=${MAX_WORKERS})…"
        $PYTHON "$HELPERS" run-agents \
            --out-dir     "$OUT_DIR" \
            --model       "$STRONG_MODEL" \
            --role        strong \
            --skills-dir  "$WORKING_SKILLS_DIR" \
            --max-turns   "$MAX_TURNS" \
            --max-input-tokens-per-turn "$MAX_INPUT_TOKENS_PER_TURN" \
            --max-workers "$MAX_WORKERS" \
            --case-ids    "${_STRONG_REMAINING[@]}" \
            $REQUIRE_SKILL $MULTI_SKILL $STRONG_REASON_ARG
        ok "Strong agent attempt ${_STRONG_ATTEMPT} complete"

        # Re-eval ALL cases (cheap — local file comparison, no LLM cost).
        # Cases that passed in earlier attempts still have their outputs in
        # cases/<id>/strong_outputs/ untouched, so they remain passing.
        step "Evaluating strong outputs (attempt ${_STRONG_ATTEMPT})…"
        $PYTHON "$HELPERS" eval \
            --dataset-dir "$DATASET_DIR" \
            --out-dir     "$OUT_DIR" \
            --iter        -1 \
            --eval-out    "$STRONG_EVAL" \
            --case-ids    "${CASE_IDS[@]}"

        # Find cases that still failed → next-round targets.
        _STRONG_REMAINING=( $($PYTHON -c "
import json, sys
d = json.load(open(sys.argv[1]))
for cid in sys.argv[2:]:
    if not d.get(cid, {}).get('pass'):
        print(cid)
" "$STRONG_EVAL" "${CASE_IDS[@]}") )

        _PASSED_NOW=$(( ${#CASE_IDS[@]} - ${#_STRONG_REMAINING[@]} ))
        ok "After attempt ${_STRONG_ATTEMPT}: ${_PASSED_NOW}/${#CASE_IDS[@]} cases passing"

        _STRONG_ATTEMPT=$(( _STRONG_ATTEMPT + 1 ))
    done

    if [[ ${#_STRONG_REMAINING[@]} -gt 0 ]]; then
        warn "Strong agent: ${#_STRONG_REMAINING[@]} case(s) still failing after ${STRONG_MAX_ATTEMPTS} attempts (excluded from downstream pool): ${_STRONG_REMAINING[*]}"
    fi
fi

STRONG_PASS_COUNT="$(count_pass "$STRONG_EVAL" "${CASE_IDS[@]}")"
ok "Strong eval: ${STRONG_PASS_COUNT} passed → ${STRONG_EVAL}"
_print_step_cost "strong-agent" "${CASES_ROOT}/**/cost_strong.json"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Batch loop — weak agent, gap agent, iterative improvement
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE_INT=${BATCH_SIZE:-${#CASE_IDS[@]}}
TOTAL_CASES=${#CASE_IDS[@]}
TOTAL_BATCHES=$(( (TOTAL_CASES + BATCH_SIZE_INT - 1) / BATCH_SIZE_INT ))

SKILL_NAMES=()
ALL_SKILL_NAMES=()  # accumulated union of skills across all batches (persists after reset)
BATCH_IDX=0
OFFSET=0
BATCHES_WITH_TRAINING=0
BASELINE_SNAPPED=0  # skill_baseline_* taken once before any patching

banner "Step 3 — Batch Loop  (${TOTAL_CASES} cases, batch size ${BATCH_SIZE_INT}, ${TOTAL_BATCHES} batch(es))"

while [[ $OFFSET -lt $TOTAL_CASES ]]; do
    BATCH_CASE_IDS=("${CASE_IDS[@]:$OFFSET:$BATCH_SIZE_INT}")
    BATCH_DIR="${OUT_DIR}/batch_$(printf '%02d' $BATCH_IDX)"
    BATCH_CASES_ROOT="${BATCH_DIR}/cases"   # per-batch: weak outputs, structures, validator reports
    mkdir -p "$BATCH_DIR" "$BATCH_CASES_ROOT"

    banner "Batch $((BATCH_IDX + 1)) / ${TOTAL_BATCHES}  [${BATCH_CASE_IDS[*]}]"

    # Each batch is independent: fresh gap evidence and skill detection
    SKILL_NAMES=()
    PREV_GAP_TRAJ_FILES=()
    PREV_GAP_NARRATIVE_FILES=()

    REMAINING_IDS=()
    NO_REMAINING=0
    _no_patch_streak=0   # consecutive iterations with no skill patch

    # ── Iterative loop ────────────────────────────────────────────────────────
    for iter in $(seq 0 "$MAX_EDITS"); do
        ITER_STR=$(printf '%02d' "$iter")
        ITER_DIR="${BATCH_DIR}/iter_${ITER_STR}"
        # Log dirs — defined unconditionally so the resume-skip branch (which
        # doesn't execute the gap step) can still reference them when collecting
        # prev-iter evidence further below.
        DIAG_LOG_DIR="${ITER_DIR}/diagnoser"
        PATCH_LOG_DIR="${ITER_DIR}/patcher"
        mkdir -p "$ITER_DIR"

        step "── Batch $((BATCH_IDX + 1)) / ${TOTAL_BATCHES}  Iteration ${iter} / ${MAX_EDITS} ──"

        # ── Validation eval (iter_1+, at iteration start with previous iter's skills) ──
        if [[ -n "$VAL_DIR" ]] && [[ "$iter" -gt 0 ]] && [[ ${#SKILL_NAMES[@]} -gt 0 ]]; then
            _do_val_eval "b$(printf '%02d' $BATCH_IDX)_iter_${ITER_STR}" "$_val_n"
            _val_n=$(( _val_n + 1 ))
        fi

        # ── (a) Run weak agent ────────────────────────────────────────────────
        # iter 0: run on all batch cases
        # iter > 0: run on all remaining cases (strong-pass pool, fixed after iter 0)
        if [[ "$iter" -eq 0 ]]; then
            RUN_CASE_IDS=("${BATCH_CASE_IDS[@]}")
        else
            RUN_CASE_IDS=("${REMAINING_IDS[@]}")
        fi

        WEAK_EVAL_ITER="${BATCH_DIR}/weak_eval_iter${ITER_STR}.json"

        WEAK_RUN_IDS=("${RUN_CASE_IDS[@]}")
        if [[ -n "$RESUME" ]]; then
            WEAK_RUN_IDS=()
            while IFS= read -r _cid; do [[ -n "$_cid" ]] && WEAK_RUN_IDS+=("$_cid"); done \
                < <(_missing_cases "$iter" "weak_trajectory.md" "${RUN_CASE_IDS[@]}")
        fi
        if [[ ${#WEAK_RUN_IDS[@]} -eq 0 ]]; then
            warn "Skipping weak agent iter ${iter} (trajectories already exist)"
        else
            [[ ${#WEAK_RUN_IDS[@]} -lt ${#RUN_CASE_IDS[@]} ]] && \
                warn "Resuming weak agent iter ${iter}: ${#WEAK_RUN_IDS[@]} / ${#RUN_CASE_IDS[@]} cases missing trajectories"
            step "Running weak agent (iter ${iter}) on ${#WEAK_RUN_IDS[@]} cases…"
            $PYTHON "$HELPERS" run-agents \
                --out-dir     "$OUT_DIR" \
                --batch-dir   "$BATCH_DIR" \
                --model       "$WEAK_MODEL" \
                --role        weak \
                --iter        "$iter" \
                --skills-dir  "$WORKING_SKILLS_DIR" \
                --max-turns   "$MAX_TURNS" \
                --max-input-tokens-per-turn "$MAX_INPUT_TOKENS_PER_TURN" \
                --max-workers "$MAX_WORKERS" \
                --case-ids    "${WEAK_RUN_IDS[@]}" \
                $REQUIRE_SKILL $MULTI_SKILL $WEAK_REASON_ARG
            ok "Weak agent (iter ${iter}) runs complete"
        fi

        if [[ -n "$RESUME" && -f "$WEAK_EVAL_ITER" ]]; then
            warn "Skipping weak eval iter ${iter} (${WEAK_EVAL_ITER} exists)"
        else
            step "Evaluating weak outputs (iter ${iter})…"
            $PYTHON "$HELPERS" eval \
                --dataset-dir "$DATASET_DIR" \
                --out-dir     "$OUT_DIR" \
                --batch-dir   "$BATCH_DIR" \
                --iter        "$iter" \
                --eval-out    "$WEAK_EVAL_ITER" \
                --case-ids    "${RUN_CASE_IDS[@]}"
        fi

        WEAK_PASS_COUNT="$(count_pass "$WEAK_EVAL_ITER" "${RUN_CASE_IDS[@]}")"
        ok "Weak eval (iter ${iter}): ${WEAK_PASS_COUNT} / ${#RUN_CASE_IDS[@]} passed"
        _print_step_cost "weak-agent iter${ITER_STR}" "${BATCH_CASES_ROOT}/**/iter_${ITER_STR}/cost_weak.json"

        # ── (b) Filter remaining cases + detect skills (iter 0 only) ─────────
        # Remaining = all batch cases where the strong agent passed.
        # This pool is fixed for the entire batch; it never shrinks even when
        # the weak agent starts passing cases in later iterations.
        if [[ "$iter" -eq 0 ]]; then
            REMAINING_IDS_FILE="${BATCH_DIR}/remaining_case_ids.txt"

            if [[ -n "$RESUME" && -f "$REMAINING_IDS_FILE" ]]; then
                warn "Skipping filter (${REMAINING_IDS_FILE} exists)"
            else
                step "Filtering: removing strong-fail cases…"
                $PYTHON "$HELPERS" filter \
                    --strong-eval "$STRONG_EVAL" \
                    --weak-eval   "$WEAK_EVAL_ITER" \
                    --case-ids    "${BATCH_CASE_IDS[@]}" \
                    > "$REMAINING_IDS_FILE"
            fi

            REMAINING_IDS=()
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -n "$line" ]] && REMAINING_IDS+=("$line")
            done < "$REMAINING_IDS_FILE"
            ok "${#REMAINING_IDS[@]} remaining cases → ${REMAINING_IDS_FILE}"

            if [[ ${#REMAINING_IDS[@]} -eq 0 ]]; then
                warn "No remaining cases in batch $((BATCH_IDX + 1)) (all strong-fail) — skipping to next batch."
                NO_REMAINING=1
                break
            fi

            echo "  Remaining cases: ${REMAINING_IDS[*]}"
            BATCHES_WITH_TRAINING=$((BATCHES_WITH_TRAINING + 1))

            # Detect skills for this batch (per-batch; batches are independent)
            WEAK_SKILL_NAMES_FILE="${BATCH_DIR}/detected_skills_weak.txt"
            STRONG_SKILL_NAMES_FILE="${BATCH_DIR}/detected_skills_strong.txt"
            SKILL_NAMES_FILE="${BATCH_DIR}/detected_skills.txt"

            if [[ -n "$RESUME" && -s "$SKILL_NAMES_FILE" ]]; then
                warn "Skipping detect-skills for batch $((BATCH_IDX + 1)) (${SKILL_NAMES_FILE} exists)"
            else
                step "Detecting skills used by weak agent (batch $((BATCH_IDX + 1)))…"
                $PYTHON "$HELPERS" detect-skills \
                    --out-dir  "$OUT_DIR" \
                    --batch-dir "$BATCH_DIR" \
                    --role     weak \
                    --iter     0 \
                    --case-ids "${REMAINING_IDS[@]}" \
                    > "$WEAK_SKILL_NAMES_FILE"

                step "Detecting skills used by strong agent (batch $((BATCH_IDX + 1)))…"
                $PYTHON "$HELPERS" detect-skills \
                    --out-dir  "$OUT_DIR" \
                    --role     strong \
                    --case-ids "${REMAINING_IDS[@]}" \
                    > "$STRONG_SKILL_NAMES_FILE"

                # Union of weak + strong skills (sorted, deduplicated)
                sort -u "$WEAK_SKILL_NAMES_FILE" "$STRONG_SKILL_NAMES_FILE" \
                    > "$SKILL_NAMES_FILE"

                ok "Weak skills:   $(paste -sd, "$WEAK_SKILL_NAMES_FILE")"
                ok "Strong skills: $(paste -sd, "$STRONG_SKILL_NAMES_FILE")"
            fi

            SKILL_NAMES=()
            while IFS= read -r line || [[ -n "$line" ]]; do
                [[ -n "$line" ]] && SKILL_NAMES+=("$line")
            done < "$SKILL_NAMES_FILE"
            if [[ ${#SKILL_NAMES[@]} -eq 0 ]]; then
                warn "No skills detected — weak agent did not call activate_skill"
                warn "Tip: check ${BATCH_DIR}/cases/<id>/iter_00/weak_skills_used.json"
                exit 1
            fi
            ok "Detected skills (union): ${SKILL_NAMES[*]}"
            # Merge into ALL_SKILL_NAMES (accumulates across batches; used for post-loop saves)
            for _s in "${SKILL_NAMES[@]}"; do
                [[ " ${ALL_SKILL_NAMES[*]:-} " == *" $_s "* ]] || ALL_SKILL_NAMES+=("$_s")
            done

            for skill in "${SKILL_NAMES[@]}"; do
                [[ -f "${WORKING_SKILLS_DIR}/${skill}/SKILL.md" ]] || {
                    fail "SKILL.md not found for '${skill}': ${WORKING_SKILLS_DIR}/${skill}/SKILL.md"
                    exit 1
                }
                # Snapshot baseline once (before any batch patches the skill).
                # On --resume, skip if already exists — the baseline is the iter-0 state
                # and must not be overwritten with the current (post-patch) skill.
                if [[ "$BASELINE_SNAPPED" -eq 0 ]]; then
                    if [[ -d "${OUT_DIR}/skill_baseline_${skill}" ]]; then
                        warn "Baseline already exists — skipping (resume): ${OUT_DIR}/skill_baseline_${skill}/"
                    else
                        _snapshot_skill "${SKILLS_DIR}/${skill}" "${OUT_DIR}/skill_baseline_${skill}"
                        ok "Baseline saved: ${OUT_DIR}/skill_baseline_${skill}/"
                    fi
                fi
            done
            BASELINE_SNAPPED=1

            # Validation eval for iter_0 — baseline skills (run after skill detection
            # so SKILL_NAMES is known; skills are unchanged at this point in iter_0)
            if [[ -n "$VAL_DIR" ]] && [[ ${#SKILL_NAMES[@]} -gt 0 ]]; then
                _do_val_eval "b$(printf '%02d' $BATCH_IDX)_iter_${ITER_STR}" "$_val_n"
                _val_n=$(( _val_n + 1 ))
            fi
        fi

        # ── (c) Abstract trajectories ─────────────────────────────────────────
        ABSTRACT_IDS=("${REMAINING_IDS[@]}")
        if [[ -n "$RESUME" ]]; then
            ABSTRACT_IDS=()
            while IFS= read -r _cid; do [[ -n "$_cid" ]] && ABSTRACT_IDS+=("$_cid"); done \
                < <(_missing_cases "$iter" "weak_structure.json" "${REMAINING_IDS[@]}")
        fi
        if [[ ${#ABSTRACT_IDS[@]} -eq 0 ]]; then
            warn "Skipping abstract iter ${iter} (all per-case structures exist)"
        else
            [[ ${#ABSTRACT_IDS[@]} -lt ${#REMAINING_IDS[@]} ]] && \
                warn "Resuming abstract iter ${iter}: ${#ABSTRACT_IDS[@]} / ${#REMAINING_IDS[@]} cases missing structures"
            step "Abstracting trajectories for ${#ABSTRACT_IDS[@]} cases (iter ${iter})…"
            $PYTHON "$HELPERS" abstract \
                --out-dir     "$OUT_DIR" \
                --batch-dir   "$BATCH_DIR" \
                --iter-dir    "$ITER_DIR" \
                --iter        "$iter" \
                --model       "$ABSTRACTOR_MODEL" \
                --max-workers "$MAX_WORKERS" \
                --case-ids    "${ABSTRACT_IDS[@]}" \
                $ABST_REASON_ARG
            ok "Abstractions saved under ${ITER_DIR}/structures/"
        fi
        _print_step_cost "abstract iter${ITER_STR}" "${BATCH_CASES_ROOT}/**/iter_${ITER_STR}/cost_abstract.json"

        # ── (d) Validate ──────────────────────────────────────────────────────
        VALIDATE_IDS=("${REMAINING_IDS[@]}")
        if [[ -n "$RESUME" ]]; then
            VALIDATE_IDS=()
            while IFS= read -r _cid; do [[ -n "$_cid" ]] && VALIDATE_IDS+=("$_cid"); done \
                < <(_missing_cases "$iter" "validator/validation_report.json" "${REMAINING_IDS[@]}")
        fi
        if [[ ${#VALIDATE_IDS[@]} -eq 0 ]]; then
            warn "Skipping validate iter ${iter} (all reports exist)"
        else
            [[ ${#VALIDATE_IDS[@]} -lt ${#REMAINING_IDS[@]} ]] && \
                warn "Resuming validate iter ${iter}: ${#VALIDATE_IDS[@]} / ${#REMAINING_IDS[@]} cases missing reports"
            step "Validating weak outputs (iter ${iter}) for ${#VALIDATE_IDS[@]} cases…"
            $PYTHON "$HELPERS" validate \
                --out-dir     "$OUT_DIR" \
                --batch-dir   "$BATCH_DIR" \
                --dataset-dir "$DATASET_DIR" \
                --iter        "$iter" \
                --model       "$VALIDATOR_MODEL" \
                --max-workers "$MAX_WORKERS" \
                --case-ids    "${VALIDATE_IDS[@]}" \
                $VAL_REASON_ARG
            ok "Validator feedback written to batch_NN/cases/<id>/iter_${ITER_STR}/validator/"
            _print_step_cost "validator iter${ITER_STR}" "${BATCH_CASES_ROOT}/**/iter_${ITER_STR}/cost_validator.json"
        fi


        # ── (e) Update case history ───────────────────────────────────────────
        step "Updating case history (iter ${iter})…"
        $PYTHON "$HELPERS" update-history \
            --batch-dir "$BATCH_DIR" \
            --out-dir   "$OUT_DIR" \
            --iter      "$iter" \
            --case-ids  "${REMAINING_IDS[@]}"

        # ── (f) Check all-pass ───────────────────────────────────────────────
        PASS_COUNT="$(count_pass "$WEAK_EVAL_ITER" "${REMAINING_IDS[@]}")"
        ok "Remaining cases (iter ${iter}): ${PASS_COUNT} / ${#REMAINING_IDS[@]} passed"

        if all_pass "$WEAK_EVAL_ITER" "${REMAINING_IDS[@]}"; then
            echo -e "\n  ${BOLD}${GREEN}✓ All ${#REMAINING_IDS[@]} remaining cases PASS on iter ${iter}${RESET}"
            break
        fi

        if [[ "$iter" -ge "$MAX_EDITS" ]]; then
            warn "Max edits (${MAX_EDITS}) reached — stopping."
            break
        fi

        # ── (g) Run GapDiagnoser, then SkillPatcher ──
        # Prefer JSON trajectory (richer, same format as case trajectories); fall back to md

        # Resume-skip check: uses the
        # Patcher completion sentinel as the canonical "this iter ran" marker.
        _gap_already_done=0
        if [[ -n "$RESUME" ]]; then
                [[ -f "${ITER_DIR}/patcher/patcher_complete" ]] && _gap_already_done=1
        fi

        if [[ "$_gap_already_done" -eq 1 ]]; then
                warn "Skipping split-framework iter ${iter} (patcher_complete exists)"
            # Restore _no_patch_streak from skill_before/skill_after snapshots so the
            # early-stop condition (2 consecutive no-patch iterations) works correctly
            # on resume.  If either snapshot is absent, assume patched (reset streak)
            # to avoid a false early-stop.
            _resume_patched=0
            for _sk in "${SKILL_NAMES[@]}"; do
                _bdir="${ITER_DIR}/skill_before_${_sk}"
                _adir="${ITER_DIR}/skill_after_${_sk}"
                if [[ ! -d "$_bdir" ]] || [[ ! -d "$_adir" ]]; then
                    _resume_patched=1; break   # snapshot missing → assume patched
                fi
                if [[ "$(_dir_hash "$_bdir")" != "$(_dir_hash "$_adir")" ]]; then
                    _resume_patched=1; break
                fi
            done
            if [[ "$_resume_patched" -eq 1 ]]; then
                _no_patch_streak=0
            else
                _no_patch_streak=$(( _no_patch_streak + 1 ))
                if [[ "$_no_patch_streak" -ge 2 ]]; then
                    warn "Early stop (resumed): no patch in ${_no_patch_streak} consecutive iterations — stopping."
                    break
                fi
            fi
            unset _resume_patched _sk _bdir _adir
        else
            for skill in "${SKILL_NAMES[@]}"; do
                _snapshot_skill "${WORKING_SKILLS_DIR}/${skill}" "${ITER_DIR}/skill_before_${skill}"
            done

            step "Running GapAgent — batch $((BATCH_IDX+1)), iter ${iter} (${#REMAINING_IDS[@]} cases)…"

            PREV_GAP_TRAJ_ARG=""
            [[ ${#PREV_GAP_TRAJ_FILES[@]} -gt 0 ]] && \
                PREV_GAP_TRAJ_ARG="--prev-gap-trajectory-files ${PREV_GAP_TRAJ_FILES[*]}"

            # Snapshot skill hashes before gap agent runs
            _SKILL_HASH_BEFORE_KEYS=()
            _SKILL_HASH_BEFORE_VALS=()
            for skill in "${SKILL_NAMES[@]}"; do
                _SKILL_HASH_BEFORE_KEYS+=("$skill")
                _SKILL_HASH_BEFORE_VALS+=("$(_dir_hash "${WORKING_SKILLS_DIR}/${skill}")")
            done

                # ── Split framework: Diagnoser → Patcher ──────────────────────
                DIAG_LOG_DIR="${ITER_DIR}/diagnoser"
                PATCH_LOG_DIR="${ITER_DIR}/patcher"
                mkdir -p "$DIAG_LOG_DIR" "$PATCH_LOG_DIR"

                # ── Diagnoser with retry ──────────────────────────────────────
                _DIAG_ATTEMPT=0
                _DIAG_MAX_RETRIES=2
                while true; do
                    _DIAG_EXIT=0
                    _run_with_timeout "$DIAGNOSER_TIMEOUT" \
                    $PYTHON "${SCRIPT_DIR}/skill_transfer/run_gap_diagnoser.py" \
                        --cases-dir               "$BATCH_CASES_ROOT" \
                        --global-cases-dir        "$GLOBAL_CASES_ROOT" \
                        --iter                    "$iter" \
                        --skill-dir               "$WORKING_SKILLS_DIR" \
                        --skill-names             "${SKILL_NAMES[@]}" \
                        --model                   "$DIAGNOSER_MODEL" \
                        --log-dir                 "$DIAG_LOG_DIR" \
                        --instruction-save-path   "${DIAG_LOG_DIR}/gap_diagnoser_instruction.md" \
                        --cost-file               "${ITER_DIR}/cost_diagnoser.json" \
                        --case-history-file       "${BATCH_DIR}/case_history.json" \
                        --max-turns               50 \
                        --trajectory-format       "$TRAJECTORY_FORMAT" \
                        --min-gap-count           "$MIN_GAP_COUNT" \
                        $PREV_GAP_TRAJ_ARG \
                        $GAP_REASON_ARG \
                        $GAP_MAX_TOK_ARG || _DIAG_EXIT=$?

                    # Diagnoser incompleteness: three signals
                    # but checks gap_report.json + diagnoser_complete sentinel.
                    _RAW_LOG="${DIAG_LOG_DIR}/gap_diagnoser_raw_log.txt"
                    _diag_incomplete=0
                    if [[ "$_DIAG_EXIT" -ne 0 ]]; then
                        _diag_incomplete=1
                    elif [[ ! -f "${DIAG_LOG_DIR}/gap_report.json" ]]; then
                        _diag_incomplete=1
                    elif [[ ! -f "${DIAG_LOG_DIR}/diagnoser_complete" ]]; then
                        _diag_incomplete=1
                    elif [[ -f "$_RAW_LOG" ]] && grep -q "response\.incomplete" "$_RAW_LOG"; then
                        _diag_incomplete=1
                    fi

                    if [[ "$_diag_incomplete" -eq 0 ]]; then
                        break
                    fi

                    _DIAG_ATTEMPT=$(( _DIAG_ATTEMPT + 1 ))
                    if [[ "$_DIAG_ATTEMPT" -gt "$_DIAG_MAX_RETRIES" ]]; then
                        warn "Diagnoser incomplete after ${_DIAG_MAX_RETRIES} retries (exit=${_DIAG_EXIT}) — skipping Patcher this iter."
                        break
                    fi
                    warn "Diagnoser incomplete (attempt ${_DIAG_ATTEMPT}/${_DIAG_MAX_RETRIES}, exit=${_DIAG_EXIT}) — resuming via session checkpoint…"

                    _ATTEMPT_ARCHIVE="${DIAG_LOG_DIR}/attempt_${_DIAG_ATTEMPT}"
                    mkdir -p "$_ATTEMPT_ARCHIVE"
                    for _f in gap_diagnoser_trajectory.md gap_diagnoser_trajectory.json \
                              gap_diagnoser_instruction.md; do
                        [[ -f "${DIAG_LOG_DIR}/${_f}" ]] && \
                            cp "${DIAG_LOG_DIR}/${_f}" "${_ATTEMPT_ARCHIVE}/${_f}"
                    done
                    [[ -f "${DIAG_LOG_DIR}/gap_diagnoser_raw_log.txt" ]] && \
                        mv "${DIAG_LOG_DIR}/gap_diagnoser_raw_log.txt" \
                           "${_ATTEMPT_ARCHIVE}/gap_diagnoser_raw_log.txt"
                    # diagnoser_session.sqlite preserved as resume source
                done

                # Only run Patcher if Diagnoser actually produced a gap_report.
                if [[ -f "${DIAG_LOG_DIR}/gap_report.json" ]]; then
                    _print_step_cost "diagnoser iter${ITER_STR}" "${ITER_DIR}/cost_diagnoser.json"

                    # ── Patcher with retry ────────────────────────────────────
                    _PATCH_ATTEMPT=0
                    _PATCH_MAX_RETRIES=2
                    while true; do
                        _PATCH_EXIT=0
                        _run_with_timeout "$PATCHER_TIMEOUT" \
                        $PYTHON "${SCRIPT_DIR}/skill_transfer/run_skill_patcher.py" \
                            --gap-report-path       "${DIAG_LOG_DIR}/gap_report.json" \
                            --cases-dir             "$BATCH_CASES_ROOT" \
                            --global-cases-dir      "$GLOBAL_CASES_ROOT" \
                            --iter                  "$iter" \
                            --skill-dir             "$WORKING_SKILLS_DIR" \
                            --skill-names           "${SKILL_NAMES[@]}" \
                            --model                 "$PATCHER_MODEL" \
                            --log-dir               "$PATCH_LOG_DIR" \
                            --instruction-save-path "${PATCH_LOG_DIR}/skill_patcher_instruction.md" \
                            --cost-file             "${ITER_DIR}/cost_patcher.json" \
                            --max-turns             50 \
                            --trajectory-format     "$TRAJECTORY_FORMAT" \
                            --min-gap-count         "$MIN_GAP_COUNT" \
                            $GAP_REASON_ARG \
                            $GAP_MAX_TOK_ARG || _PATCH_EXIT=$?

                        # Patcher incompleteness: completion sentinel
                        # gap_patches_complete (written by finalize_patches tool)
                        # OR patcher_complete (runner success path).
                        _RAW_LOG="${PATCH_LOG_DIR}/skill_patcher_raw_log.txt"
                        _patch_incomplete=0
                        if [[ "$_PATCH_EXIT" -ne 0 ]]; then
                            _patch_incomplete=1
                        elif [[ ! -f "${PATCH_LOG_DIR}/gap_patches_complete" ]]; then
                            _patch_incomplete=1
                        elif [[ ! -f "${PATCH_LOG_DIR}/patcher_complete" ]]; then
                            _patch_incomplete=1
                        elif [[ -f "$_RAW_LOG" ]] && grep -q "response\.incomplete" "$_RAW_LOG"; then
                            _patch_incomplete=1
                        fi

                        if [[ "$_patch_incomplete" -eq 0 ]]; then
                            break
                        fi

                        _PATCH_ATTEMPT=$(( _PATCH_ATTEMPT + 1 ))
                        if [[ "$_PATCH_ATTEMPT" -gt "$_PATCH_MAX_RETRIES" ]]; then
                            warn "Patcher incomplete after ${_PATCH_MAX_RETRIES} retries (exit=${_PATCH_EXIT}) — continuing with partially patched skill."
                            break
                        fi
                        warn "Patcher incomplete (attempt ${_PATCH_ATTEMPT}/${_PATCH_MAX_RETRIES}, exit=${_PATCH_EXIT}) — resuming via session checkpoint…"

                        _ATTEMPT_ARCHIVE="${PATCH_LOG_DIR}/attempt_${_PATCH_ATTEMPT}"
                        mkdir -p "$_ATTEMPT_ARCHIVE"
                        for _f in skill_patcher_trajectory.md skill_patcher_trajectory.json \
                                  skill_patcher_instruction.md; do
                            [[ -f "${PATCH_LOG_DIR}/${_f}" ]] && \
                                cp "${PATCH_LOG_DIR}/${_f}" "${_ATTEMPT_ARCHIVE}/${_f}"
                        done
                        [[ -f "${PATCH_LOG_DIR}/skill_patcher_raw_log.txt" ]] && \
                            mv "${PATCH_LOG_DIR}/skill_patcher_raw_log.txt" \
                               "${_ATTEMPT_ARCHIVE}/skill_patcher_raw_log.txt"
                        # patcher_session.sqlite preserved as resume source
                    done
                    unset _PATCH_EXIT _PATCH_ATTEMPT _PATCH_MAX_RETRIES _patch_incomplete
                fi
                unset _DIAG_EXIT _DIAG_ATTEMPT _DIAG_MAX_RETRIES _diag_incomplete _RAW_LOG _ATTEMPT_ARCHIVE

            # Check which skills were actually modified
            _any_patched=0
            for skill in "${SKILL_NAMES[@]}"; do
                _snapshot_skill "${WORKING_SKILLS_DIR}/${skill}" "${ITER_DIR}/skill_after_${skill}"
                _hash_after="$(_dir_hash "${WORKING_SKILLS_DIR}/${skill}")"
                
                _before_hash=""
                for i in "${!_SKILL_HASH_BEFORE_KEYS[@]}"; do
                    if [[ "${_SKILL_HASH_BEFORE_KEYS[$i]}" == "$skill" ]]; then
                        _before_hash="${_SKILL_HASH_BEFORE_VALS[$i]}"
                        break
                    fi
                done

                if [[ "$_before_hash" != "$_hash_after" ]]; then
                    ok "Skill '${skill}' patched."
                    _any_patched=1
                else
                    warn "Skill '${skill}' unchanged (no systemic gaps or patch skipped)."
                fi
            done
            unset _SKILL_HASH_BEFORE_KEYS _SKILL_HASH_BEFORE_VALS _before_hash _hash_after
                [[ "$_any_patched" -eq 1 ]] && ok "Split-framework logs: ${DIAG_LOG_DIR} + ${PATCH_LOG_DIR}" \
                                            || warn "Split-framework logs: ${DIAG_LOG_DIR} + ${PATCH_LOG_DIR}"
                _print_step_cost "patcher iter${ITER_STR}" "${ITER_DIR}/cost_patcher.json"

            # Early stop: 2 consecutive iterations with no patch
            if [[ "$_any_patched" -eq 1 ]]; then
                _no_patch_streak=0
            else
                _no_patch_streak=$(( _no_patch_streak + 1 ))
                if [[ "$_no_patch_streak" -ge 2 ]]; then
                    warn "Early stop: no skill patch in ${_no_patch_streak} consecutive iterations — stopping."
                    break
                fi
            fi
        fi

        # Prev-iter evidence: the Diagnoser holds gap_report.json (+ trajectory)
        # and the Patcher writes gap_patch_narrative.json next to it, so the next
        # iteration's --prev-gap-patch-narrative-files resolves the sibling report.
            _DIAG_TRAJ_JSON="${DIAG_LOG_DIR}/gap_diagnoser_trajectory.json"
            _DIAG_TRAJ_MD="${DIAG_LOG_DIR}/gap_diagnoser_trajectory.md"
            if [[ -f "$_DIAG_TRAJ_JSON" ]]; then
                PREV_GAP_TRAJ_FILES+=("$_DIAG_TRAJ_JSON")
            elif [[ -f "$_DIAG_TRAJ_MD" ]]; then
                PREV_GAP_TRAJ_FILES+=("$_DIAG_TRAJ_MD")
            fi
            # Patcher narrative lives one level deeper but uses the same
            # filename — Diagnoser's --prev-gap-patch-narrative-files resolves
            # the sibling gap_report.json from the same dir, so we point at the
            # Diagnoser dir which has gap_report.json AND we point separately at
            # the Patcher narrative.  Since the Diagnoser tool expects
            # gap_patch_narrative.json + gap_report.json in the SAME dir, copy
            # the Patcher narrative next to the Diagnoser's gap_report.json so
            # next iter's lite-mode evidence resolves correctly.
            _PATCHER_NARRATIVE="${PATCH_LOG_DIR}/gap_patch_narrative.json"
            if [[ -f "$_PATCHER_NARRATIVE" ]]; then
                cp "$_PATCHER_NARRATIVE" "${DIAG_LOG_DIR}/gap_patch_narrative.json"
                PREV_GAP_NARRATIVE_FILES+=("${DIAG_LOG_DIR}/gap_patch_narrative.json")
            fi
            unset _DIAG_TRAJ_JSON _DIAG_TRAJ_MD _PATCHER_NARRATIVE

    done  # end iter loop

    OFFSET=$((OFFSET + BATCH_SIZE_INT))
    BATCH_IDX=$((BATCH_IDX + 1))
done

# ── Reasoning-effort downgrade summary ───────────────────────────────────────
# Scan all batch_*/iter_*/{diagnoser,patcher}/effort_downgrade.json
# files written by the Python runners when they had to lower reasoning effort
# in response to `response.incomplete` truncation.  Print one row per agent
# that experienced a downgrade.
echo
echo -e "${BOLD}${CYAN}── Reasoning-effort downgrade summary ──────────────────────${RESET}"
$PYTHON - "$OUT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
rows = []
for batch_dir in sorted(out_dir.glob("batch_*")):
    if not batch_dir.is_dir(): continue
    for iter_dir in sorted(batch_dir.glob("iter_*")):
        if not iter_dir.is_dir(): continue
        for agent_sub in ["diagnoser", "patcher"]:
            f = iter_dir / agent_sub / "effort_downgrade.json"
            if not f.exists(): continue
            try:
                history = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not history: continue
            final = history[-1].get("new_effort", "?")
            rows.append({
                "batch":   batch_dir.name,
                "iter":    iter_dir.name,
                "agent":   agent_sub,
                "events":  len(history),
                "final":   final,
                "first":   history[0].get("new_effort", "?"),
            })
if not rows:
    print("  (no reasoning-effort downgrades occurred)")
else:
    print(f"  {'batch':<10} {'iter':<10} {'agent':<12} {'events':>7}  {'first→final effort'}")
    for r in rows:
        ladder = (r["first"] if r["first"] == r["final"] else f"{r['first']} → {r['final']}")
        print(f"  {r['batch']:<10} {r['iter']:<10} {r['agent']:<12} {r['events']:>7}  {ladder}")
PYEOF

# ── Best-skill selection (no extra val eval — last iter's val eval is already current) ──
if [[ -n "$VAL_DIR" ]] && [[ ${#ALL_SKILL_NAMES[@]} -gt 0 ]]; then

    # Print summary table and save the best skill snapshot
    $PYTHON - "${VAL_OUT_DIR}/val_eval_summary.json" "$OUT_DIR" "$VAL_OUT_DIR" <<'PYEOF'
import json, sys, shutil, os

summary_file = sys.argv[1]
out_dir      = sys.argv[2]
val_out_dir  = sys.argv[3]

BOLD="\033[1m"; GREEN="\033[32m"; RESET="\033[0m"; CYAN="\033[36m"; DIM="\033[90m"

if not os.path.exists(summary_file):
    print("  No val_eval_summary.json found — skipping best-skill selection")
    sys.exit(0)

data = json.load(open(summary_file))
if not data:
    sys.exit(0)

# Print table
col = max(len(k) for k in data) + 2
print(f"\n{BOLD}{CYAN}── Validation pass rate by iteration {'─'*32}{RESET}")
for label, info in data.items():
    marker = " ◀ best" if label == max(data, key=lambda k: data[k]["pass_rate"]) else ""
    print(f"  {label:<{col}}  {info['passed']}/{info['total']}  ({info['pass_rate']:.1%}){marker}")

best_label = max(data, key=lambda k: data[k]["pass_rate"])
best_info  = data[best_label]
print(f"\n  {BOLD}Best checkpoint: {best_label} — "
      f"{best_info['passed']}/{best_info['total']} ({best_info['pass_rate']:.1%}){RESET}")

# Copy best snapshot → skill_best_<skill>/
snap_dir = os.path.join(val_out_dir, f"skill_snap_{best_label}")
if os.path.isdir(snap_dir):
    for skill in os.listdir(snap_dir):
        src = os.path.join(snap_dir, skill)
        dst = os.path.join(out_dir, f"skill_best_{skill}")
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  {GREEN}✓{RESET} Best skill saved: {dst}")
else:
    print(f"  {DIM}Warning: snapshot not found at {snap_dir}{RESET}")

# Persist best-skill metadata
json.dump({"best_iter": best_label, **best_info},
          open(os.path.join(out_dir, "best_skill_info.json"), "w"), indent=2)
PYEOF
fi

# ── Save final skill states ───────────────────────────────────────────────────
if [[ ${#ALL_SKILL_NAMES[@]} -gt 0 ]]; then
    for skill in "${ALL_SKILL_NAMES[@]}"; do
        _snapshot_skill "${WORKING_SKILLS_DIR}/${skill}" "${OUT_DIR}/skill_final_${skill}"
        ok "Final skill saved: ${OUT_DIR}/skill_final_${skill}/"
        diff -ru "${OUT_DIR}/skill_baseline_${skill}" "${OUT_DIR}/skill_final_${skill}" \
            > "${OUT_DIR}/skill_changes_${skill}.diff" || true
        ok "Skill diff saved: ${OUT_DIR}/skill_changes_${skill}.diff"
    done

else
    warn "No skills were patched (no training cases found in any batch)."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════
banner "Pipeline Complete"

echo -e "  Total cases      : ${TOTAL_CASES}"
echo -e "  Batches processed: ${TOTAL_BATCHES}  (${BATCHES_WITH_TRAINING} with training cases)"
echo -e "  Max edits/batch  : ${MAX_EDITS}"
echo -e "  Skills patched   : ${ALL_SKILL_NAMES[*]:-none}"
echo -e "  Output dir       : ${OUT_DIR}"
if [[ -n "$VAL_DIR" ]]; then
    BEST_LABEL="$($PYTHON - "${VAL_OUT_DIR}/val_eval_summary.json" 2>/dev/null <<'PYEOF'
import json, sys, os
f = sys.argv[1]
if os.path.exists(f):
    d = json.load(open(f))
    best = max(d, key=lambda k: d[k]["pass_rate"]) if d else "n/a"
    info = d.get(best, {})
    print(f"{best}  ({info.get('passed','?')}/{info.get('total','?')} = {info.get('pass_rate',0):.1%})")
PYEOF
)"
    echo -e "  Val best iter    : ${BEST_LABEL}"
    echo -e "  Best skill dir   : ${OUT_DIR}/skill_best_<skill>/"
fi

# ── Cost aggregation ──────────────────────────────────────────────────────────
TOTAL_COST_FILE="${OUT_DIR}/cost_total.json"
$PYTHON - "$OUT_DIR" "$TOTAL_COST_FILE" <<'PYEOF'
import json, sys, glob, os

out_dir    = sys.argv[1]
total_file = sys.argv[2]

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[36m"; GREEN = "\033[32m"; DIM = "\033[90m"

cost_files = sorted(
    f for f in glob.glob(os.path.join(out_dir, "**", "cost_*.json"), recursive=True)
    if os.path.basename(f) != "cost_total.json"
)
if not cost_files:
    print("  (no cost files found)")
    sys.exit(0)

runs = []
for cf in cost_files:
    try:
        data = json.load(open(cf))
        if "runs" in data:
            runs.extend(data["runs"])
        elif "label" in data:
            runs.append(data)
    except Exception:
        pass

if not runs:
    print("  (no cost data found)")
    sys.exit(0)

col_w = max(len(r.get("label") or r.get("model", "?")) for r in runs) + 2
print(f"\n{BOLD}{CYAN}── Cost Breakdown {'─'*44}{RESET}")
for r in runs:
    name   = (r.get("label") or r.get("model", "?")).ljust(col_w)
    in_t   = r.get("input_tokens", 0)
    cache_t= r.get("cached_input_tokens", 0)
    out_t  = r.get("output_tokens", 0)
    cost   = r.get("total_cost_usd", 0.0)
    print(f"  {name}  {in_t:>8,} in  {cache_t:>8,} cached  {out_t:>8,} out  {GREEN}${cost:.4f}{RESET}")

total_in     = sum(r.get("input_tokens",        0)   for r in runs)
total_cached = sum(r.get("cached_input_tokens", 0)   for r in runs)
total_out    = sum(r.get("output_tokens",       0)   for r in runs)
total_cost   = sum(r.get("total_cost_usd",      0.0) for r in runs)
print(f"  {'─'*(col_w+52)}")
print(f"  {'TOTAL'.ljust(col_w)}  {total_in:>8,} in  {total_cached:>8,} cached  {total_out:>8,} out  {BOLD}{GREEN}${total_cost:.4f}{RESET}")

summary = {
    "runs": runs,
    "total_input_tokens":        total_in,
    "total_cached_input_tokens": total_cached,
    "total_output_tokens":       total_out,
    "total_tokens":              total_in + total_out,
    "total_cost_usd":            round(total_cost, 6),
}
json.dump(summary, open(total_file, "w"), indent=2)
print(f"\n  {DIM}Saved: {total_file}{RESET}")
PYEOF

echo
