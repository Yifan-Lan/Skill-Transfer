#!/usr/bin/env bash
#
# Evaluate one agent + one skill on a dataset split.  No strong agent, no gap
# analysis — just run the agent over every case and score the outputs.
#
# With --pass-at-n N the agent is run N times and Pass@k is reported for k=1..N.
#
# Usage:
#   export OPENAI_API_KEY=...            # or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
#   bash run_eval.sh \
#       --dataset-dir data/spreadsheetbench/test_200 \
#       --out-dir     runs/eval_sb_adapted \
#       --skills-dir  runs/sb_gpt41mini/skill_best_xlsx \
#       --model       gpt-4.1-mini
#
# Pass --no-skill to measure the agent without any skill at all.
# Run `bash run_eval.sh --help` for the full flag list.

# Activate the conda env (Python 3.11 + LibreOffice). See setup_env.sh for overrides.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env.sh"
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.11}"
HELPERS="${SCRIPT_DIR}/skill_transfer/pipeline_helpers.py"

DATASET_DIR=""
OUT_DIR=""
SKILLS_DIR="${SCRIPT_DIR}/skills"
WEAK_MODEL="gpt-4.1-mini"
MAX_WORKERS=4
MAX_TURNS=30
MAX_INPUT_TOKENS_PER_TURN=100000000
SIMPLE_ONLY=""
RESUME=""
MAX_SAMPLES=""
SPLIT=""                  # optional split name (e.g. probe_50); forwarded to prepare
SPLIT_MANIFEST=""         # optional explicit split_manifest.json path
NO_BIRD_SCHEMA=""         # "--no-bird-schema" agentic BIRD setup (agent discovers schema)
REQUIRE_SKILL=""          # "--require-skill" flag forwarded to run-agents
MULTI_SKILL=""            # "--multi-skill"   flag forwarded to run-agents
NO_SKILL=""               # "--no-skill" clean control: drop skill mechanism entirely
WEAK_REASONING_EFFORT=""  # reasoning effort for weak agent (low/medium/high, empty=off)
PASS_AT_N=1               # number of independent agent samples; Pass@k computed for k=1..N
BEST_MODE=""              # if "1", report Pass@k as best-of-k union (max over all k-of-N subsets) to mitigate sample noise
REF_TRAJ_DIR=""           # optional: dir whose cases/<id>/iter_00/weak_trajectory.json are injected as reference

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)   DATASET_DIR="$2";              shift 2 ;;
        --out-dir)       OUT_DIR="$2";                  shift 2 ;;
        --skills-dir)    SKILLS_DIR="$2";               shift 2 ;;
        --weak-model)    WEAK_MODEL="$2";               shift 2 ;;
        --max-workers)   MAX_WORKERS="$2";              shift 2 ;;
        --max-turns)                  MAX_TURNS="$2";                          shift 2 ;;
        --max-input-tokens-per-turn)  MAX_INPUT_TOKENS_PER_TURN="$2";          shift 2 ;;
        --simple-only)                SIMPLE_ONLY="--simple-only";             shift 1 ;;
        --resume)        RESUME="1";                   shift 1 ;;
        --max-samples)   MAX_SAMPLES="$2";              shift 2 ;;
        --split)         SPLIT="$2";                    shift 2 ;;
        --split-manifest) SPLIT_MANIFEST="$2";          shift 2 ;;
        --no-bird-schema) NO_BIRD_SCHEMA="--no-bird-schema"; shift 1 ;;
        --require-skill)         REQUIRE_SKILL="--require-skill";      shift 1 ;;
        --multi-skill)           MULTI_SKILL="--multi-skill";          shift 1 ;;
        --no-skill)              NO_SKILL="--no-skill";                shift 1 ;;
        --weak-reasoning-effort) WEAK_REASONING_EFFORT="$2";           shift 2 ;;
        --pass-at-n)             PASS_AT_N="$2";                        shift 2 ;;
        --best)                  BEST_MODE="1";                        shift 1 ;;
        --ref-traj-dir)          REF_TRAJ_DIR="$2";                    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$DATASET_DIR" ]] && { echo "ERROR: --dataset-dir is required" >&2; exit 1; }
[[ -z "$OUT_DIR" ]]     && { echo "ERROR: --out-dir is required"     >&2; exit 1; }
[[ -f "$DATASET_DIR/dataset.json" ]] || { echo "ERROR: dataset.json not found in $DATASET_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# ── Save current skills snapshot ──────────────────────────────────────────────
mkdir -p "${OUT_DIR}/skills_snapshot"
for skill_md in "${SKILLS_DIR}"/*/SKILL.md; do
    if [[ -f "$skill_md" ]]; then
        skill_name=$(basename "$(dirname "$skill_md")")
        cp "$skill_md" "${OUT_DIR}/skills_snapshot/${skill_name}_SKILL.md"
    fi
done

# ── Colour helpers ────────────────────────────────────────────────────────────
# (defined early so the API banner can use them)
BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RED="\033[31m"; DIM="\033[90m"
banner() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"; echo -e "  ${BOLD}$1${RESET}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${RESET}"; }
step()   { echo -e "\n${BOLD}${CYAN}▶${RESET} ${BOLD}$1${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()   { echo -e "  ${YELLOW}!${RESET} $1"; }

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



CASES_ROOT="${OUT_DIR}/cases"
CASE_IDS_FILE="${OUT_DIR}/case_ids.txt"
COST_TOTAL_FILE="${OUT_DIR}/cost_total.json"

# Reasoning effort arg string (empty when not set)
WEAK_REASON_ARG=""; [[ -n "$WEAK_REASONING_EFFORT" ]] && WEAK_REASON_ARG="--reasoning-effort $WEAK_REASONING_EFFORT"
# Reference trajectory dir arg string (empty when not set)
REF_TRAJ_ARG=""; [[ -n "$REF_TRAJ_DIR" ]] && REF_TRAJ_ARG="--ref-traj-dir $REF_TRAJ_DIR"

# ── Per-case resume helper ─────────────────────────────────────────────────────
# Prints (one per line) the case IDs whose iter_<iter_str>/<rel_path> file is absent.
_missing_cases() {
    local iter_str="$1"; local rel="$2"; shift 2
    for cid in "$@"; do
        [[ -f "${CASES_ROOT}/${cid}/iter_${iter_str}/${rel}" ]] || echo "$cid"
    done
}

# ── Step cost printer ─────────────────────────────────────────────────────────
_print_step_cost() {
    local label="$1"; shift
    local patterns=("$@")
    $PYTHON - "$label" "${patterns[@]}" <<'PYEOF'
import json, sys, glob, os
BOLD="\033[1m"; RESET="\033[0m"; GREEN="\033[32m"; DIM="\033[90m"
label    = sys.argv[1]
patterns = sys.argv[2:]
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

# ══════════════════════════════════════════════════════════════════════════════
banner "Step 1 — Prepare Cases"
# ══════════════════════════════════════════════════════════════════════════════

if [[ -n "$RESUME" && -s "$CASE_IDS_FILE" ]]; then
    warn "Skipping prepare (--resume, ${CASE_IDS_FILE} exists)"
else
    step "Preparing case directories from dataset…"
    _SPLIT_ARGS=()
    [[ -n "$SPLIT" ]]          && _SPLIT_ARGS+=(--split "$SPLIT")
    [[ -n "$SPLIT_MANIFEST" ]] && _SPLIT_ARGS+=(--split-manifest "$SPLIT_MANIFEST")
    [[ -n "$NO_BIRD_SCHEMA" ]] && _SPLIT_ARGS+=("$NO_BIRD_SCHEMA")
    $PYTHON "$HELPERS" prepare \
        --dataset-dir "$DATASET_DIR" \
        --out-dir     "$OUT_DIR" \
        "${_SPLIT_ARGS[@]}" \
        $SIMPLE_ONLY \
        > "$CASE_IDS_FILE"
fi

CASE_IDS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] && CASE_IDS+=("$line")
done < "$CASE_IDS_FILE"

if [[ -n "$MAX_SAMPLES" && ${#CASE_IDS[@]} -gt "$MAX_SAMPLES" ]]; then
    CASE_IDS=("${CASE_IDS[@]:0:$MAX_SAMPLES}")
    warn "Limited to $MAX_SAMPLES cases (--max-samples)"
fi

ok "${#CASE_IDS[@]} cases prepared → ${CASES_ROOT}"

[[ ${#CASE_IDS[@]} -eq 0 ]] && { echo -e "  ${RED}✗${RESET} No cases found"; exit 1; }

# ── Pass@n: run one complete sample loop per attempt ──────────────────────────
# For PASS_AT_N=1 the loop executes once with the same file names as before.
for _SAMPLE in $(seq 0 $((PASS_AT_N - 1))); do
    _ITER_STR=$(printf "%02d" "$_SAMPLE")

    # File names for this sample
    if [[ "$PASS_AT_N" -eq 1 ]]; then
        _EVAL_PRE="${OUT_DIR}/eval_pre_recalc.json"
        _EVAL_POST="${OUT_DIR}/eval.json"
        _BANNER_SFX=""
    else
        _EVAL_PRE="${OUT_DIR}/eval_pre_sample_${_ITER_STR}.json"
        _EVAL_POST="${OUT_DIR}/eval_sample_${_ITER_STR}.json"
        _BANNER_SFX=" · sample $((_SAMPLE + 1))/${PASS_AT_N}"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    banner "Step 2 — Run Weak Agent  (${WEAK_MODEL})${_BANNER_SFX}"
    # ══════════════════════════════════════════════════════════════════════════

    RUN_IDS=("${CASE_IDS[@]}")
    if [[ -n "$RESUME" ]]; then
        RUN_IDS=()
        while IFS= read -r _cid; do [[ -n "$_cid" ]] && RUN_IDS+=("$_cid"); done \
            < <(_missing_cases "$_ITER_STR" "weak_trajectory.json" "${CASE_IDS[@]}")
        # fall back to .md if no .json (older runs)
        if [[ ${#RUN_IDS[@]} -gt 0 ]]; then
            RUN_IDS=()
            while IFS= read -r _cid; do [[ -n "$_cid" ]] && RUN_IDS+=("$_cid"); done \
                < <(_missing_cases "$_ITER_STR" "weak_trajectory.md" "${CASE_IDS[@]}")
        fi
    fi

    if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
        warn "Skipping weak agent${_BANNER_SFX} (all ${#CASE_IDS[@]} trajectories already exist)"
    else
        [[ ${#RUN_IDS[@]} -lt ${#CASE_IDS[@]} ]] && \
            warn "Resuming: ${#RUN_IDS[@]} / ${#CASE_IDS[@]} cases still need trajectories"
        step "Running weak agent on ${#RUN_IDS[@]} cases${_BANNER_SFX} (max_workers=${MAX_WORKERS})…"
        $PYTHON "$HELPERS" run-agents \
            --out-dir     "$OUT_DIR" \
            --model       "$WEAK_MODEL" \
            --role        weak \
            --iter        "$_SAMPLE" \
            --skills-dir  "$SKILLS_DIR" \
            --max-turns                   "$MAX_TURNS" \
            --max-input-tokens-per-turn   "$MAX_INPUT_TOKENS_PER_TURN" \
            --max-workers                 "$MAX_WORKERS" \
            --case-ids                    "${RUN_IDS[@]}" \
            $REQUIRE_SKILL $MULTI_SKILL $NO_SKILL $WEAK_REASON_ARG $REF_TRAJ_ARG
        ok "Agent runs complete"
        _print_step_cost "weak-agent${_BANNER_SFX}" "${CASES_ROOT}/**/iter_${_ITER_STR}/cost_weak.json"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    banner "Step 3 — Evaluate (Pre-Recalc)${_BANNER_SFX}"
    # ══════════════════════════════════════════════════════════════════════════

    if [[ -n "$RESUME" && -f "$_EVAL_PRE" ]]; then
        warn "Skipping pre-recalc eval${_BANNER_SFX} (${_EVAL_PRE} exists)"
    else
        step "Evaluating outputs before formula recalculation${_BANNER_SFX}…"
        $PYTHON "$HELPERS" eval \
            --dataset-dir "$DATASET_DIR" \
            --out-dir     "$OUT_DIR" \
            --iter        "$_SAMPLE" \
            --eval-out    "$_EVAL_PRE" \
            --case-ids    "${CASE_IDS[@]}"
    fi

    # ══════════════════════════════════════════════════════════════════════════
    banner "Step 4 — Recalculate Formulas (LibreOffice)${_BANNER_SFX}"
    # ══════════════════════════════════════════════════════════════════════════

    step "Backing up pre-recalc outputs${_BANNER_SFX}…"
    $PYTHON - "$OUT_DIR" "$_SAMPLE" "${CASE_IDS[@]}" <<'PYEOF'
import sys, shutil
from pathlib import Path
out_dir  = Path(sys.argv[1])
iter_n   = int(sys.argv[2])
case_ids = sys.argv[3:]
copied = skipped = 0
for cid in case_ids:
    src = out_dir / "cases" / cid / f"iter_{iter_n:02d}" / "weak_outputs"
    dst = out_dir / "cases" / cid / f"iter_{iter_n:02d}" / "weak_outputs_pre_recalc"
    if not src.is_dir():
        continue
    if dst.exists():
        skipped += 1
        continue
    shutil.copytree(str(src), str(dst))
    copied += 1
print(f"  Backed up {copied} cases → weak_outputs_pre_recalc  ({skipped} already existed)")
PYEOF

    step "Opening output workbooks in LibreOffice to cache formula values${_BANNER_SFX}…"
    $PYTHON "$HELPERS" recalculate \
        --out-dir     "$OUT_DIR" \
        --iter        "$_SAMPLE" \
        --max-workers "$MAX_WORKERS" \
        --case-ids    "${CASE_IDS[@]}"
    ok "Recalculation complete"

    # ══════════════════════════════════════════════════════════════════════════
    banner "Step 5 — Evaluate (Post-Recalc)${_BANNER_SFX}"
    # ══════════════════════════════════════════════════════════════════════════

    if [[ -n "$RESUME" && -f "$_EVAL_POST" ]]; then
        warn "Skipping post-recalc eval${_BANNER_SFX} (${_EVAL_POST} exists)"
    else
        step "Evaluating outputs after formula recalculation${_BANNER_SFX}…"
        $PYTHON "$HELPERS" eval \
            --dataset-dir "$DATASET_DIR" \
            --out-dir     "$OUT_DIR" \
            --iter        "$_SAMPLE" \
            --eval-out    "$_EVAL_POST" \
            --case-ids    "${CASE_IDS[@]}"
    fi

done  # end sample loop

# ══════════════════════════════════════════════════════════════════════════════
banner "Step 6 — Results"
# ══════════════════════════════════════════════════════════════════════════════

$PYTHON - "$OUT_DIR" "$PASS_AT_N" "$WEAK_MODEL" "${BEST_MODE:-0}" "${CASE_IDS[@]}" <<'PYEOF'
import json, sys, itertools
from pathlib import Path

out_dir    = Path(sys.argv[1])
pass_at_n  = int(sys.argv[2])
model_name = sys.argv[3]
best_mode  = sys.argv[4] == "1"
case_ids   = sys.argv[5:]
n_total    = len(case_ids)

RESET  = "\033[0m"; BOLD = "\033[1m"
GREEN  = "\033[32m"; RED  = "\033[31m"; YELLOW = "\033[33m"
CYAN   = "\033[36m"; DIM  = "\033[90m"

def eval_pre_path(s):
    if pass_at_n == 1:
        return out_dir / "eval_pre_recalc.json"
    return out_dir / f"eval_pre_sample_{s:02d}.json"

def eval_post_path(s):
    if pass_at_n == 1:
        return out_dir / "eval.json"
    return out_dir / f"eval_sample_{s:02d}.json"

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}

data_pre  = [load(eval_pre_path(s))  for s in range(pass_at_n)]
data_post = [load(eval_post_path(s)) for s in range(pass_at_n)]

def bar_str(n, total, width=30):
    fill = int(width * n / total) if total else 0
    return f"{GREEN}{'█' * fill}{RESET}{DIM}{'░' * (width - fill)}{RESET}"

# ── Pass@n = 1: original pre/post table ───────────────────────────────────────
if pass_at_n == 1:
    pre_passed  = [cid for cid in case_ids if data_pre[0].get(cid,  {}).get("pass")]
    pre_failed  = [cid for cid in case_ids if not data_pre[0].get(cid,  {}).get("pass")]
    post_passed = [cid for cid in case_ids if data_post[0].get(cid, {}).get("pass")]
    post_failed = [cid for cid in case_ids if not data_post[0].get(cid, {}).get("pass")]

    pct_pre  = 100 * len(pre_passed)  / n_total if n_total else 0
    pct_post = 100 * len(post_passed) / n_total if n_total else 0
    delta    = len(post_passed) - len(pre_passed)
    delta_str = (f"{GREEN}+{delta}{RESET}" if delta > 0
                 else (f"{RED}{delta}{RESET}" if delta < 0 else f"{DIM}0{RESET}"))

    print(f"\n  Model  : {BOLD}{model_name}{RESET}   Cases: {n_total}\n")
    print(f"  {'':20s}  {'Pass':>5}   {'Fail':>5}   {'Rate':>6}   Bar")
    print(f"  {'─'*70}")
    print(f"  {'Pre-recalc':20s}  {BOLD}{GREEN}{len(pre_passed):>5}{RESET}   {BOLD}{RED}{len(pre_failed):>5}{RESET}   {pct_pre:>5.1f}%   {bar_str(len(pre_passed), n_total)}")
    print(f"  {'Post-recalc':20s}  {BOLD}{GREEN}{len(post_passed):>5}{RESET}   {BOLD}{RED}{len(post_failed):>5}{RESET}   {pct_post:>5.1f}%   {bar_str(len(post_passed), n_total)}")
    print(f"  {'─'*70}")
    print(f"  {'Recalc delta':20s}  {delta_str}  cases changed by LibreOffice recalculation\n")

    gained = sorted(set(post_passed) - set(pre_passed))
    lost   = sorted(set(pre_passed)  - set(post_passed))
    if gained:
        print(f"  {GREEN}Newly passing after recalc:{RESET}")
        for cid in gained:
            print(f"    {GREEN}✓{RESET} {cid}")
    if lost:
        print(f"  {RED}Newly failing after recalc:{RESET}")
        for cid in lost:
            print(f"    {RED}✗{RESET} {cid}")
    if gained or lost:
        print()

    # OfficeQA fine-grained tolerance metrics
    def is_officeqa_case(cid):
        try:
            return json.load(open(out_dir / "cases" / cid / "task_meta.json")).get("type") == "officeqa"
        except Exception:
            return False

    officeqa_cases = [cid for cid in case_ids if is_officeqa_case(cid)]
    if officeqa_cases:
        tol_keys   = ["exact", "0.1pct", "1pct", "5pct"]
        tol_labels = {"exact": "Exact (0%)", "0.1pct": "≤0.1%", "1pct": "≤1%", "5pct": "≤5%"}
        print(f"  {BOLD}{CYAN}OfficeQA fuzzy-match accuracy ({len(officeqa_cases)} cases):{RESET}")
        for tol in tol_keys:
            n = sum(1 for cid in officeqa_cases
                    if data_post[0].get(cid, {}).get("scores_by_tolerance", {}).get(tol, 0) == 1.0)
            p = 100 * n / len(officeqa_cases) if officeqa_cases else 0
            print(f"    {tol_labels[tol]:12s}  {bar_str(n, len(officeqa_cases))}  {BOLD}{n}/{len(officeqa_cases)}{RESET} ({p:.1f}%)")
        print()

    def is_searchqa_case(cid):
        try:
            return json.load(open(out_dir / "cases" / cid / "task_meta.json")).get("type") == "searchqa"
        except Exception:
            return False

    searchqa_cases = [cid for cid in case_ids if is_searchqa_case(cid)]
    if searchqa_cases:
        em = sum(float(data_post[0].get(cid, {}).get("score", 0.0))
                 for cid in searchqa_cases) / len(searchqa_cases)
        f1 = sum(float(data_post[0].get(cid, {}).get("f1", 0.0))
                 for cid in searchqa_cases) / len(searchqa_cases)
        print(f"  {BOLD}{CYAN}SearchQA metrics ({len(searchqa_cases)} cases):{RESET}")
        print(f"    EM  {BOLD}{em * 100:.1f}%{RESET}")
        print(f"    F1  {BOLD}{f1 * 100:.1f}%{RESET}\n")

    if post_failed:
        print(f"  {DIM}Failed cases (post-recalc):{RESET}")
        for cid in sorted(post_failed):
            r = data_post[0].get(cid, {})
            msg = r.get("message") or r.get("rationale", "")
            hint = f"  {DIM}{msg[:80]}{RESET}" if msg else ""
            print(f"    {RED}✗{RESET} {cid}{hint}")

    if post_passed:
        print(f"\n  {DIM}Passed cases (post-recalc):{RESET}")
        for cid in sorted(post_passed):
            print(f"    {GREEN}✓{RESET} {cid}")

    print(f"\n  {DIM}Pre-recalc results  → {eval_pre_path(0)}{RESET}")
    print(f"  {DIM}Post-recalc results → {eval_post_path(0)}{RESET}\n")

# ── Pass@n > 1: per-sample table + Pass@k + per-case pass counts ─────────────
else:
    # Per-sample pass counts (post-recalc and pre-recalc)
    post_per_sample = [sum(1 for cid in case_ids if data_post[s].get(cid, {}).get("pass"))
                       for s in range(pass_at_n)]
    pre_per_sample  = [sum(1 for cid in case_ids if data_pre[s].get(cid,  {}).get("pass"))
                       for s in range(pass_at_n)]

    print(f"\n  Model  : {BOLD}{model_name}{RESET}   Cases: {n_total}   Samples (n): {pass_at_n}\n")
    print(f"  {'Sample':12s}  {'Pre':>5}  {'Post':>5}  {'Post%':>6}   Bar")
    print(f"  {'─'*65}")
    for s in range(pass_at_n):
        pct = 100 * post_per_sample[s] / n_total if n_total else 0
        print(f"  {('Sample ' + str(s)):12s}  {pre_per_sample[s]:>5}  {post_per_sample[s]:>5}  {pct:>5.1f}%   {bar_str(post_per_sample[s], n_total)}")
    print(f"  {'─'*65}\n")

    # Pass@k table, before and after LibreOffice recalculation
    # Standard: union over the first k samples (in order)
    # Best:     max union over all k-of-N subsets (mitigates single-sample noise)
    mode_label = "best k-of-N" if best_mode else "first k samples"
    print(f"  {BOLD}{CYAN}Pass@k  (at least 1 pass in {mode_label}):{RESET}")
    print(f"  {'k':>6}  {'Pre':>12}  {'Pre%':>7}  {'Post':>12}  {'Post%':>7}")
    print(f"  {'─'*55}")
    pre_pass_at_k_counts = []
    post_pass_at_k_counts = []

    def _count_at_k(samples_data, k):
        if best_mode:
            best = 0
            for combo in itertools.combinations(range(pass_at_n), k):
                u = sum(
                    1 for cid in case_ids
                    if any(samples_data[s].get(cid, {}).get("pass") for s in combo)
                )
                if u > best:
                    best = u
            return best
        return sum(
            1 for cid in case_ids
            if any(samples_data[s].get(cid, {}).get("pass") for s in range(k))
        )

    for k in range(1, pass_at_n + 1):
        n_pre = _count_at_k(data_pre, k)
        n_post = _count_at_k(data_post, k)
        pre_pass_at_k_counts.append(n_pre)
        post_pass_at_k_counts.append(n_post)
        pct_pre = 100 * n_pre / n_total if n_total else 0
        pct_post = 100 * n_post / n_total if n_total else 0
        print(
            f"  Pass@{k:<3d}  "
            f"{BOLD}{n_pre:>5}/{n_total:<5}{RESET}  {pct_pre:>6.1f}%  "
            f"{BOLD}{n_post:>5}/{n_total:<5}{RESET}  {pct_post:>6.1f}%"
        )
    print()

    # Per-case pass count summary (post-recalc is the main metric; pre is saved in JSON)
    pre_pass_counts = {
        cid: sum(1 for s in range(pass_at_n) if data_pre[s].get(cid, {}).get("pass"))
        for cid in case_ids
    }
    pass_counts = {
        cid: sum(1 for s in range(pass_at_n) if data_post[s].get(cid, {}).get("pass"))
        for cid in case_ids
    }

    always_pass  = sorted(cid for cid in case_ids if pass_counts[cid] == pass_at_n)
    partial_pass = sorted(
        ((cid, pass_counts[cid]) for cid in case_ids if 0 < pass_counts[cid] < pass_at_n),
        key=lambda x: -x[1]
    )
    never_pass   = sorted(cid for cid in case_ids if pass_counts[cid] == 0)

    print(f"  {BOLD}Per-case pass counts  (format: case_id  passes/n){RESET}")
    print(f"  {'─'*65}")

    if always_pass:
        print(f"  {GREEN}Always pass ({len(always_pass)}/{n_total}):{RESET}")
        for cid in always_pass:
            print(f"    {GREEN}✓{RESET} {cid}  {GREEN}{pass_at_n}/{pass_at_n}{RESET}")

    if partial_pass:
        print(f"  {YELLOW}Partial pass ({len(partial_pass)}/{n_total}):{RESET}")
        for cid, cnt in partial_pass:
            bar = f"{GREEN}{'●' * cnt}{RESET}{DIM}{'○' * (pass_at_n - cnt)}{RESET}"
            print(f"    {YELLOW}~{RESET} {cid}  {bar}  {cnt}/{pass_at_n}")

    if never_pass:
        print(f"  {RED}Never pass ({len(never_pass)}/{n_total}):{RESET}")
        for cid in never_pass:
            r = data_post[-1].get(cid, {})
            msg = r.get("message") or r.get("rationale", "")
            hint = f"  {DIM}{msg[:70]}{RESET}" if msg else ""
            print(f"    {RED}✗{RESET} {cid}{hint}")

    print()

    # Save aggregated pass@n summary
    summary_path = out_dir / "eval_pass_at_n.json"
    json.dump({
        "pass_at_n": pass_at_n,
        "best_mode": best_mode,
        "model": model_name,
        "n_cases": n_total,
        "pass_at_k": {str(k): c for k, c in enumerate(post_pass_at_k_counts, 1)},
        "pass_at_k_rate": {str(k): round(c / n_total, 4) if n_total else 0
                           for k, c in enumerate(post_pass_at_k_counts, 1)},
        "pre_recalc_pass_at_k": {str(k): c for k, c in enumerate(pre_pass_at_k_counts, 1)},
        "pre_recalc_pass_at_k_rate": {str(k): round(c / n_total, 4) if n_total else 0
                                      for k, c in enumerate(pre_pass_at_k_counts, 1)},
        "post_recalc_pass_at_k": {str(k): c for k, c in enumerate(post_pass_at_k_counts, 1)},
        "post_recalc_pass_at_k_rate": {str(k): round(c / n_total, 4) if n_total else 0
                                       for k, c in enumerate(post_pass_at_k_counts, 1)},
        "per_sample_post_pass": post_per_sample,
        "per_sample_pre_pass":  pre_per_sample,
        "per_case_pass_counts": {cid: pass_counts[cid] for cid in sorted(case_ids)},
        "per_case_pre_recalc_pass_counts": {cid: pre_pass_counts[cid] for cid in sorted(case_ids)},
        "per_case_post_recalc_pass_counts": {cid: pass_counts[cid] for cid in sorted(case_ids)},
    }, open(summary_path, "w"), indent=2)
    print(f"  {DIM}Pass@n summary → {summary_path}{RESET}\n")

PYEOF

# ── Cost summary ──────────────────────────────────────────────────────────────
$PYTHON - "$OUT_DIR" "$COST_TOTAL_FILE" <<'PYEOF'
import json, sys, glob, os
BOLD="\033[1m"; RESET="\033[0m"; GREEN="\033[32m"; DIM="\033[90m"; CYAN="\033[36m"
out_dir        = sys.argv[1]
total_out_file = sys.argv[2]

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
        name = os.path.splitext(os.path.basename(cf))[0].replace("cost_", "")
        inner = data.get("runs", [data] if "total_cost_usd" in data else [])
        for r in inner:
            runs.append({**r, "_label": name})
    except Exception:
        pass

if not runs:
    print("  (no cost data found)")
    sys.exit(0)

col_w = max(len(r.get("_label", "")) for r in runs)
print(f"\n{BOLD}{CYAN}── Cost Breakdown ────────────────────────────────────────{RESET}")
total_in, total_cached, total_out, total_cost = 0, 0, 0, 0.0
for r in runs:
    name    = r.get("_label", "?")
    in_t    = r.get("input_tokens",        0)
    cache_t = r.get("cached_input_tokens", 0)
    out_t   = r.get("output_tokens",       0)
    cost    = r.get("total_cost_usd",      0.0)
    total_in += in_t; total_cached += cache_t; total_out += out_t; total_cost += cost
    print(f"  {name.ljust(col_w)}  {in_t:>8,} in  {cache_t:>8,} cached  {out_t:>6,} out  {GREEN}${cost:.4f}{RESET}")
print(f"  {'─' * (col_w + 46)}")
print(f"  {'TOTAL'.ljust(col_w)}  {total_in:>8,} in  {total_cached:>8,} cached  {total_out:>6,} out  {BOLD}{GREEN}${total_cost:.4f}{RESET}\n")

json.dump({"total_cost_usd": round(total_cost, 6), "input_tokens": total_in,
           "cached_input_tokens": total_cached, "output_tokens": total_out,
           "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]},
          open(total_out_file, "w"), indent=2)
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
banner "Step 7 — Fine-grained metrics"
# ══════════════════════════════════════════════════════════════════════════════
# Auto-detect task family from the produced eval JSON and compute the matching
# finer-grained metric (best-effort — NEVER fails the eval).  Summaries are
# written into OUT_DIR and consumed by run_ablation*.sh:
#   OfficeQA  (scores_by_tolerance present) → tolerance_summary.json  (accuracy@tol)
#   SpreadsheetBench (xlsx)                 → cell_match_summary.json  (cell match ratio)

_EVAL_JSON=""
for _cand in "${OUT_DIR}/eval.json" "${OUT_DIR}/eval_sample_00.json"; do
    [[ -f "$_cand" ]] && { _EVAL_JSON="$_cand"; break; }
done

if [[ -z "$_EVAL_JSON" ]]; then
    warn "No eval JSON found — skipping fine-grained metrics"
else
    # Detect task family from a case's task_meta.json (reliable), so WTQ is not
    # mistaken for SpreadsheetBench: officeqa | searchqa | wtq | ssbench.
    _FAMILY=$($PYTHON -c "
import json, glob
metas = sorted(glob.glob('${OUT_DIR}/cases/*/task_meta.json'))
fam = 'ssbench'
if metas:
    m = json.load(open(metas[0]))
    if m.get('type') == 'officeqa': fam = 'officeqa'
    elif m.get('type') == 'searchqa': fam = 'searchqa'
    elif m.get('type') == 'dabench': fam = 'dabench'
    elif m.get('instruction_type') == 'WikiTableQuestion': fam = 'wtq'
print(fam)
" 2>/dev/null || echo "ssbench")
    case "$_FAMILY" in
        officeqa)
            step "OfficeQA — accuracy within tolerance"
            $PYTHON "${SCRIPT_DIR}/skill_transfer/compute_tolerance_accuracy.py" --eval-dirs "$OUT_DIR" \
                || warn "tolerance-accuracy metric failed (non-fatal)" ;;
        wtq)
            step "WikiTableQuestions — denotation accuracy + breakdown"
            $PYTHON "${SCRIPT_DIR}/skill_transfer/compute_wtq_breakdown.py" --eval-dirs "$OUT_DIR" \
                || warn "wtq-breakdown metric failed (non-fatal)" ;;
        dabench)
            step "InfiAgent-DABench — accuracy by question/sub-question + concept breakdown"
            $PYTHON "${SCRIPT_DIR}/skill_transfer/compute_dabench_breakdown.py" --eval-dirs "$OUT_DIR" --dataset-dir "$DATASET_DIR" \
                || warn "dabench-breakdown metric failed (non-fatal)" ;;
        searchqa)
            step "SearchQA — normalized exact match + token F1"
            ok "SearchQA metrics are stored per case in ${_EVAL_JSON}" ;;
        *)
            step "SpreadsheetBench — cell match ratio (post-hoc re-score of saved output.xlsx)"
            $PYTHON "${SCRIPT_DIR}/skill_transfer/compute_cell_match.py" --eval-dirs "$OUT_DIR" --dataset-dir "$DATASET_DIR" \
                || warn "cell-match metric failed (non-fatal)" ;;
    esac
fi
