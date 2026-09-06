#!/usr/bin/env bash
#
# Roll a Skill Transfer run back to the start of a given iteration, so it can be
# resumed after a crash or re-run with different settings.
#
# It restores the skill files from that iteration's snapshot and deletes the
# artifacts produced at or after it (diagnoser/, patcher/, weak outputs, evals).
#
#   --iter N          iteration to roll back to
#   --after-gap       keep the gap step of iteration N; redo only what follows
#   --keep-session    keep the agents' session databases so a resumed run
#                     continues the same conversation instead of restarting
#   --dry-run         print what would change and exit
#
# Usage:
#   bash restore_run.sh --out-dir runs/sb_gpt41mini --skills-dir skills/xlsx_baseline --iter 2
set -euo pipefail

PYTHON="${PYTHON:-python3.11}"

OUT_DIR=""
RESTORE_ITER=""
SKILLS_DIR=""
BATCH=""
DRY_RUN=""
KEEP_SESSION=""
AFTER_GAP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)       OUT_DIR="$2";      shift 2 ;;
        --iter)          RESTORE_ITER="$2"; shift 2 ;;
        --skills-dir)    SKILLS_DIR="$2";   shift 2 ;;
        --batch)         BATCH="$2";        shift 2 ;;
        --after-gap)     AFTER_GAP="1";     shift 1 ;;
        --keep-session)  KEEP_SESSION="1";  shift 1 ;;
        --dry-run)       DRY_RUN="1";       shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Auto-derive SKILLS_DIR from $OUT_DIR/working_skills/ if not explicitly set.
# Matches the per-run isolation in run_skill_transfer.sh.  Honored only when
# the directory actually exists — otherwise leave SKILLS_DIR empty and let the
# user know via the dry-run / status line.
if [[ -z "$SKILLS_DIR" && -n "$OUT_DIR" && -d "${OUT_DIR}/working_skills" ]]; then
    SKILLS_DIR="${OUT_DIR}/working_skills"
fi

# Guard: if --skills-dir was given (or derived) but the directory does not
# exist, ABORT before any destructive step.  Previously a non-existent path
# (e.g. a stale macOS /Users/... path on a Linux host) only printed a per-skill
# "cannot restore" line and continued — so iter dirs got deleted while the
# skill was silently NOT restored, leaving working_skills contaminated.
if [[ -n "$SKILLS_DIR" && ! -d "$SKILLS_DIR" ]]; then
    echo "ERROR: --skills-dir does not exist: ${SKILLS_DIR}" >&2
    if [[ -n "$OUT_DIR" && -d "${OUT_DIR}/working_skills" ]]; then
        echo "       Omit --skills-dir to auto-restore to ${OUT_DIR}/working_skills," >&2
        echo "       or pass --skills-dir ${OUT_DIR}/working_skills explicitly." >&2
    else
        echo "       Pass the per-run working_skills dir (<out-dir>/working_skills)," >&2
        echo "       not the read-only canonical baseline." >&2
    fi
    exit 1
fi

if [[ -n "$AFTER_GAP" ]] && [[ -n "$KEEP_SESSION" ]]; then
    echo "ERROR: --after-gap and --keep-session are mutually exclusive" >&2
    exit 1
fi

[[ -z "$OUT_DIR" ]]      && { echo "ERROR: --out-dir is required"  >&2; exit 1; }
[[ -z "$RESTORE_ITER" ]] && { echo "ERROR: --iter is required"     >&2; exit 1; }
[[ ! -d "$OUT_DIR" ]]    && { echo "ERROR: not found: $OUT_DIR"    >&2; exit 1; }

ITER_STR=$(printf '%02d' "$RESTORE_ITER")

# ── colour helpers ────────────────────────────────────────────────────────────
BOLD="\033[1m"; RESET="\033[0m"
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; DIM="\033[90m"

banner() { echo -e "\n${BOLD}$1${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()   { echo -e "  ${YELLOW}!${RESET} $1"; }
fail()   { echo -e "  ${RED}✗${RESET} $1"; }

# del <path> — delete dir or file (or just print in dry-run mode)
del() {
    local target="$1"
    if [[ -n "$DRY_RUN" ]]; then
        if   [[ -d "$target" ]]; then echo -e "  ${YELLOW}[dry] rm -rf${RESET}  $target"
        elif [[ -f "$target" ]]; then echo -e "  ${YELLOW}[dry] rm     ${RESET}  $target"
        fi
    else
        if   [[ -d "$target" ]]; then rm -rf "$target"; ok "Deleted dir:  $target"
        elif [[ -f "$target" ]]; then rm -f  "$target"; ok "Deleted file: $target"
        fi
    fi
}

# ── collect batch directories ─────────────────────────────────────────────────
if [[ -n "$BATCH" ]]; then
    BATCH_DIRS=("${OUT_DIR}/batch_$(printf '%02d' "$BATCH")")
else
    BATCH_DIRS=()
    for d in "${OUT_DIR}"/batch_*/; do
        [[ -d "$d" ]] && BATCH_DIRS+=("${d%/}")
    done
fi

[[ ${#BATCH_DIRS[@]} -eq 0 ]] && { fail "No batch_* directories found in $OUT_DIR"; exit 1; }

# ── summary ───────────────────────────────────────────────────────────────────
if [[ -n "$AFTER_GAP" ]]; then
    banner "Restore pipeline → after gap-agent of iter ${RESTORE_ITER} (= start of iter $((RESTORE_ITER+1)))"
    echo -e "  out-dir    : $OUT_DIR"
    echo -e "  target point: AFTER gap-agent of iter_${ITER_STR}"
    echo -e "               (keeps diagnoser/ + patcher/ + cost + snapshots of iter_${ITER_STR},"
    echo -e "                deletes iter_$((RESTORE_ITER+1))+ so weak-agent re-runs from here)"
    echo -e "  batches    : ${#BATCH_DIRS[@]}  →  ${BATCH_DIRS[*]}"
    echo -e "  mode       : ${YELLOW}--after-gap${RESET} — skills restored from skill_after_* snapshots"
    [[ -n "$SKILLS_DIR" ]] && echo -e "  skills-dir : $SKILLS_DIR" \
                           || echo -e "  skills-dir : ${YELLOW}(not provided — skill will NOT be restored)${RESET}"
elif [[ -n "$KEEP_SESSION" ]]; then
    banner "Restore pipeline → iter ${RESTORE_ITER} gap-agent step (checkpoint resume)"
    echo -e "  out-dir    : $OUT_DIR"
    echo -e "  target iter: ${RESTORE_ITER}  (keeps weak-run + eval + validate of iter_${ITER_STR},"
    echo -e "               deletes gap-agent volatile logs, preserves session DB)"
    echo -e "  batches    : ${#BATCH_DIRS[@]}  →  ${BATCH_DIRS[*]}"
    echo -e "  mode       : ${YELLOW}--keep-session${RESET} — volatile logs deleted, session DB preserved"
    echo -e "               skills NOT restored (session carries patch history)"
else
    banner "Restore pipeline → iter ${RESTORE_ITER} gap-agent step (fresh start)"
    echo -e "  out-dir    : $OUT_DIR"
    echo -e "  target iter: ${RESTORE_ITER}  (keeps weak-run + eval + validate of iter_${ITER_STR},"
    echo -e "               deletes gap-agent of iter_${ITER_STR} and all of iter_$((RESTORE_ITER+1))+)"
    echo -e "  batches    : ${#BATCH_DIRS[@]}  →  ${BATCH_DIRS[*]}"
    [[ -n "$SKILLS_DIR" ]] && echo -e "  skills-dir : $SKILLS_DIR" \
                           || echo -e "  skills-dir : ${YELLOW}(not provided — skill will NOT be restored)${RESET}"
fi
[[ -n "$DRY_RUN" ]]    && echo -e "\n  ${YELLOW}${BOLD}DRY RUN — no files will be changed${RESET}"

# ── per-batch cleanup ─────────────────────────────────────────────────────────
for batch_dir in "${BATCH_DIRS[@]}"; do
    banner "── Batch: $(basename "$batch_dir") ──"

    ITER_DIR="${batch_dir}/iter_${ITER_STR}"
    if [[ ! -d "$ITER_DIR" ]]; then
        warn "iter_${ITER_STR} not found in $batch_dir — skipping batch"
        continue
    fi

    # Split-framework subdirs (present only when run_skill_transfer.sh was
    # invoked with --gap-framework split).  All cleanup branches below handle
    DIAG_DIR="${ITER_DIR}/diagnoser"
    PATCH_DIR="${ITER_DIR}/patcher"
    DIAG_SESSION="${DIAG_DIR}/diagnoser_session.sqlite"
    PATCH_SESSION="${PATCH_DIR}/patcher_session.sqlite"

    if [[ -n "$AFTER_GAP" ]]; then
        # ── (1a) --after-gap: restore skills from skill_after_* snapshots ────
        # Target point is AFTER the gap step completed, so skills should be in
        # their post-gap state.  skill_after_* holds that snapshot.
        # are kept (gap is done; no need to re-run it).
        # In split mode: diagnoser/, patcher/, cost_diagnoser.json,
        # cost_patcher.json, and snapshots are kept.
        if [[ -n "$SKILLS_DIR" ]]; then
            found_snapshot=0
            for snapshot in "${ITER_DIR}"/skill_after_*; do
                [[ -d "$snapshot" ]] || continue
                skill_name="$(basename "$snapshot")"
                skill_name="${skill_name#skill_after_}"
                skill_live="${SKILLS_DIR}/${skill_name}"
                if [[ -n "$DRY_RUN" ]]; then
                    echo -e "  ${YELLOW}[dry] restore skill '${skill_name}' (post-gap):${RESET}"
                    echo -e "        ${snapshot}/"
                    echo -e "        → ${skill_live}/"
                else
                    if [[ -d "$skill_live" ]]; then
                        rm -rf "${skill_live:?}"
                        cp -r "$snapshot" "$skill_live"
                        ok "Restored skill '${skill_name}'  ←  iter_${ITER_STR}/skill_after_${skill_name}/"
                    else
                        fail "Skill dir not found: $skill_live — cannot restore"
                    fi
                fi
                found_snapshot=1
            done
            [[ "$found_snapshot" -eq 0 ]] && warn "No skill_after_*/ found in iter_${ITER_STR} — skill NOT restored"
        fi
        # Report what is kept based on which framework's artifacts exist.
        _kept_parts=()
        [[ -d "$DIAG_DIR" ]]      && _kept_parts+=("${DIAG_DIR}/  cost_diagnoser.json")
        [[ -d "$PATCH_DIR" ]]     && _kept_parts+=("${PATCH_DIR}/  cost_patcher.json")
        _kept_parts+=("skill snapshots")
        [[ -n "$DRY_RUN" ]] \
            && echo -e "  ${YELLOW}[dry] keep ${_kept_parts[*]}${RESET}" \
            || ok "Kept: ${_kept_parts[*]}"
        unset _kept_parts

    elif [[ -n "$KEEP_SESSION" ]]; then
        # ── (1b) --keep-session: wipe volatile logs, preserve session DBs ────
        # Skill files NOT restored — the session DB carries the agent's patch
        # history; restoring would create a memory/disk inconsistency.


        # ─ Split-framework diagnoser/ ─
        if [[ -d "$DIAG_DIR" ]]; then
            if [[ ! -f "$DIAG_SESSION" ]]; then
                warn "--keep-session: diagnoser session DB not found (${DIAG_SESSION})"
            else
                ok "--keep-session: diagnoser session DB preserved  ($(du -sh "$DIAG_SESSION" 2>/dev/null | cut -f1))"
            fi
            for _f in \
                "${DIAG_DIR}/gap_report.json" \
                "${DIAG_DIR}/gap_report_draft.json" \
                "${DIAG_DIR}/gap_report_pre_decompose.json" \
                "${DIAG_DIR}/gap_diagnoser_trajectory.md" \
                "${DIAG_DIR}/gap_diagnoser_trajectory.json" \
                "${DIAG_DIR}/gap_diagnoser_raw_log.txt" \
                "${DIAG_DIR}/gap_diagnoser_instruction.md" \
                "${DIAG_DIR}/gap_diagnoser_system_prompt.md" \
                "${DIAG_DIR}/diagnoser_complete" \
                "${DIAG_DIR}/gap_patch_narrative.json" \
                "${DIAG_DIR}/effort_downgrade.json" \
                ; do
                [[ -e "$_f" ]] && del "$_f"
            done
            for _d in "${DIAG_DIR}"/attempt_*/; do
                [[ -d "$_d" ]] && del "${_d%/}"
            done
            del "${ITER_DIR}/cost_diagnoser.json"
        fi

        # ─ Split-framework patcher/ ─
        if [[ -d "$PATCH_DIR" ]]; then
            if [[ ! -f "$PATCH_SESSION" ]]; then
                warn "--keep-session: patcher session DB not found (${PATCH_SESSION})"
            else
                ok "--keep-session: patcher session DB preserved  ($(du -sh "$PATCH_SESSION" 2>/dev/null | cut -f1))"
            fi
            for _f in \
                "${PATCH_DIR}/skill_patcher_trajectory.md" \
                "${PATCH_DIR}/skill_patcher_trajectory.json" \
                "${PATCH_DIR}/skill_patcher_raw_log.txt" \
                "${PATCH_DIR}/skill_patcher_instruction.md" \
                "${PATCH_DIR}/skill_patcher_system_prompt.md" \
                "${PATCH_DIR}/patcher_complete" \
                "${PATCH_DIR}/gap_patches_complete" \
                "${PATCH_DIR}/gap_patch_narrative.json" \
                "${PATCH_DIR}/skill_edits.diff" \
                "${PATCH_DIR}/effort_downgrade.json" \
                ; do
                [[ -e "$_f" ]] && del "$_f"
            done
            for _d in "${PATCH_DIR}"/attempt_*/; do
                [[ -d "$_d" ]] && del "${_d%/}"
            done
            del "${ITER_DIR}/cost_patcher.json"
        fi

        [[ ! -d "$DIAG_DIR" ]] && [[ ! -d "$PATCH_DIR" ]] && \
            warn "No diagnoser/ or patcher/ found in iter_${ITER_STR} — nothing to clean"

    else
        # ── (1c) Full restore: skills from skill_before_*, delete gap step dirs ─
        # MUST run before step (2) deletes the skill_before_*/ directories.
        if [[ -n "$SKILLS_DIR" ]]; then
            found_snapshot=0
            for snapshot in "${ITER_DIR}"/skill_before_*; do
                [[ -d "$snapshot" ]] || continue
                skill_name="$(basename "$snapshot")"
                skill_name="${skill_name#skill_before_}"
                skill_live="${SKILLS_DIR}/${skill_name}"
                if [[ -n "$DRY_RUN" ]]; then
                    echo -e "  ${YELLOW}[dry] restore skill '${skill_name}':${RESET}"
                    echo -e "        ${snapshot}/"
                    echo -e "        → ${skill_live}/"
                else
                    if [[ -d "$skill_live" ]]; then
                        rm -rf "${skill_live:?}"
                        cp -r "$snapshot" "$skill_live"
                        ok "Restored skill '${skill_name}'  ←  iter_${ITER_STR}/skill_before_${skill_name}/"
                    else
                        fail "Skill dir not found: $skill_live — cannot restore"
                    fi
                fi
                found_snapshot=1
            done
            [[ "$found_snapshot" -eq 0 ]] && warn "No skill_before_*/ found in iter_${ITER_STR} — skill NOT restored"
        fi

        # Full delete: remove the diagnoser/ and patcher/ dirs.
        if [[ -f "$DIAG_SESSION" ]] && [[ -z "$DRY_RUN" ]]; then
            warn "diagnoser session DB found — deleting (use --keep-session to preserve)"
        fi
        if [[ -f "$PATCH_SESSION" ]] && [[ -z "$DRY_RUN" ]]; then
            warn "patcher session DB found — deleting (use --keep-session to preserve)"
        fi
        [[ -d "$DIAG_DIR" ]]      && del "$DIAG_DIR"
        [[ -d "$PATCH_DIR" ]]     && del "$PATCH_DIR"
        del "${ITER_DIR}/cost_diagnoser.json"
        del "${ITER_DIR}/cost_patcher.json"

        for d in "${ITER_DIR}"/skill_before_* "${ITER_DIR}"/skill_after_*; do
            [[ -e "$d" ]] && del "$d"
        done
    fi

    # ── (3) Remove batch iter N+1, N+2, … dirs and their eval files ──────────
    next_iter=$((RESTORE_ITER + 1))
    while true; do
        next_str=$(printf '%02d' "$next_iter")
        found=0
        [[ -d "${batch_dir}/iter_${next_str}" ]]               && { del "${batch_dir}/iter_${next_str}";               found=1; }
        [[ -f "${batch_dir}/weak_eval_iter${next_str}.json" ]] && { del "${batch_dir}/weak_eval_iter${next_str}.json"; found=1; }
        [[ "$found" -eq 0 ]] && break
        next_iter=$((next_iter + 1))
    done

    # ── (4) Remove per-case iter N+1, N+2, … directories ─────────────────────
    # Weak-specific data lives under batch_NN/cases/{id}/iter_NN/ (per-batch layout)
    BATCH_CASES_ROOT="${batch_dir}/cases"
    if [[ -d "$BATCH_CASES_ROOT" ]]; then
        next_iter=$((RESTORE_ITER + 1))
        while true; do
            next_str=$(printf '%02d' "$next_iter")
            found=0
            for case_dir in "${BATCH_CASES_ROOT}"/*/; do
                [[ -d "$case_dir" ]] || continue
                target="${case_dir%/}/iter_${next_str}"
                if [[ -d "$target" ]]; then
                    del "$target"
                    found=1
                fi
            done
            [[ "$found" -eq 0 ]] && break
            next_iter=$((next_iter + 1))
        done
    fi

    # ── (5) Truncate case_history.json ───────────────────────────────────────
    history_file="${batch_dir}/case_history.json"
    if [[ -f "$history_file" ]]; then
        if [[ -n "$DRY_RUN" ]]; then
            echo -e "  ${YELLOW}[dry] truncate case_history.json — keep iter_history[iter <= ${RESTORE_ITER}]${RESET}"
        else
            $PYTHON - "$history_file" "$RESTORE_ITER" <<'PYEOF'
import json, sys
path, keep_through = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    history = json.load(f)

removed = 0
for case_data in history.get("cases", {}).values():
    before = case_data.get("iter_history", [])
    after  = [e for e in before if int(e["iter"]) <= keep_through]
    removed += len(before) - len(after)
    case_data["iter_history"] = after

history["updated_at_iter"] = keep_through

with open(path, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2, ensure_ascii=False)
print(f"    case_history.json: removed {removed} iter_history entries (iter > {keep_through})")
PYEOF
            ok "Truncated: $history_file"
        fi
    else
        warn "case_history.json not found — nothing to truncate"
    fi
done

# ── val_evals cleanup ─────────────────────────────────────────────────────────
# Val eval labels use the format b{NN}_iter_{NN} (e.g. b00_iter_01).
# Cleanup is scoped to the batches being restored: other batches' entries are kept.
#
# ORDER: val_cases cleanup (a) MUST run before the eval-file deletion (b).
# The Python script in (a) globs weak_val_*.json to build the label→_val_n map;
# deleting those files first (old bug) would leave the map empty.
VAL_OUT_DIR="${OUT_DIR}/val_evals"
if [[ -d "$VAL_OUT_DIR" ]]; then
    banner "── Val-evals cleanup ──"

    # Build list of affected batch index strings (e.g. "00" "01")
    AFFECTED_BATCH_STRS=()
    for _bd in "${BATCH_DIRS[@]}"; do
        _b="$(basename "$_bd")"; AFFECTED_BATCH_STRS+=("${_b#batch_}")
    done

    # Collect labels to delete BEFORE touching any files, so (a) can still glob them.
    LABELS_TO_DELETE=()
    for _bstr in "${AFFECTED_BATCH_STRS[@]}"; do
        next_iter=$((RESTORE_ITER + 1))
        while true; do
            next_str=$(printf '%02d' "$next_iter")
            label="b${_bstr}_iter_${next_str}"
            if [[ -f "${VAL_OUT_DIR}/weak_val_${label}.json" ]] || \
               [[ -d "${VAL_OUT_DIR}/skill_snap_${label}" ]]; then
                LABELS_TO_DELETE+=("$label")
                next_iter=$((next_iter + 1))
            else
                break
            fi
        done
    done

    # (a) Delete per-case iter dirs inside val_evals/cases/ BEFORE deleting eval files.
    #     Val case trajectories live at iter_{_val_n}/ where _val_n is a global
    #     monotonic counter.  Python resolves label→_val_n by sorting all currently-
    #     present weak_val_b*_iter_*.json files — must run while they still exist.
    VAL_CASES_ROOT="${VAL_OUT_DIR}/cases"
    if [[ -d "$VAL_CASES_ROOT" ]] && [[ ${#LABELS_TO_DELETE[@]} -gt 0 ]]; then
        if [[ -n "$DRY_RUN" ]]; then
            echo -e "  ${YELLOW}[dry] val_evals/cases/: would delete iter dirs for: ${LABELS_TO_DELETE[*]}${RESET}"
        else
            $PYTHON - "$VAL_OUT_DIR" "$VAL_CASES_ROOT" "${LABELS_TO_DELETE[@]}" <<'PYEOF'
import re, shutil, sys
from pathlib import Path

val_out_dir    = Path(sys.argv[1])
val_cases_root = Path(sys.argv[2])
labels_to_del  = set(sys.argv[3:])   # e.g. {"b00_iter_02", "b00_iter_03"}

# Build val_n_map from ALL currently-present weak_val_b*_iter_*.json files.
pat = re.compile(r'^weak_val_(b(\d+)_iter_(\d+))\.json$')
all_labels = []
for f in val_out_dir.glob('weak_val_b*_iter_*.json'):
    m = pat.match(f.name)
    if m:
        all_labels.append((m.group(1), int(m.group(2)), int(m.group(3))))
# Sort by (batch_idx, iter_idx) = the order _val_n was assigned
all_labels.sort(key=lambda x: (x[1], x[2]))
val_n_map = {lbl: idx for idx, (lbl, _, _) in enumerate(all_labels)}

to_delete_val_n = sorted(
    val_n_map[lbl] for lbl in labels_to_del if lbl in val_n_map
)

deleted = 0
for val_n in to_delete_val_n:
    iter_str = f"iter_{val_n:02d}"
    for case_dir in val_cases_root.iterdir():
        target = case_dir / iter_str
        if target.is_dir():
            shutil.rmtree(target)
            deleted += 1
if to_delete_val_n:
    print(f"    val_evals/cases/: deleted {deleted} iter dirs (val_n={to_delete_val_n})")
PYEOF
        fi
    fi

    # (b) Now delete weak_val_b{NN}_iter_{K}.json and skill_snap_b{NN}_iter_{K}/
    if [[ ${#LABELS_TO_DELETE[@]} -gt 0 ]]; then
        for label in "${LABELS_TO_DELETE[@]}"; do
            [[ -f "${VAL_OUT_DIR}/weak_val_${label}.json" ]] && del "${VAL_OUT_DIR}/weak_val_${label}.json"
            [[ -d "${VAL_OUT_DIR}/skill_snap_${label}"    ]] && del "${VAL_OUT_DIR}/skill_snap_${label}"
        done
    else
        warn "No val eval labels found to delete for affected batches (iter > ${RESTORE_ITER})"
    fi

    # (c) Truncate val_eval_summary.json — remove affected-batch entries where iter > RESTORE_ITER
    #     Entries for unaffected batches are preserved unchanged.
    summary_file="${VAL_OUT_DIR}/val_eval_summary.json"
    if [[ -f "$summary_file" ]]; then
        if [[ -n "$DRY_RUN" ]]; then
            echo -e "  ${YELLOW}[dry] truncate val_eval_summary.json — keep affected-batch iter <= ${RESTORE_ITER}${RESET}"
        else
            $PYTHON - "$summary_file" "$RESTORE_ITER" "${AFFECTED_BATCH_STRS[@]}" <<'PYEOF'
import json, re, sys

path         = sys.argv[1]
keep_through = int(sys.argv[2])
affected     = set(sys.argv[3:])   # batch index strings e.g. {"00", "01"}

with open(path, encoding="utf-8") as f:
    summary = json.load(f)

pat = re.compile(r'^b(\d+)_iter_(\d+)$')

def keep(label):
    m = pat.match(label)
    if not m:
        return True                          # unrecognised format — keep
    if m.group(1) not in affected:
        return True                          # different batch — keep
    return int(m.group(2)) <= keep_through   # same batch: keep only if iter <= N

kept    = {k: v for k, v in summary.items() if keep(k)}
removed = len(summary) - len(kept)

with open(path, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)
print(f"    val_eval_summary.json: removed {removed} entries (affected batches, iter > {keep_through})")
PYEOF
            ok "Truncated: $summary_file"
        fi
    fi
fi

# ── top-level terminal outputs (stale if pipeline ran to completion) ──────────
# Includes best-skill selection artifacts: skill_best_*/ and best_skill_info.json
# point at a winning iter that this rollback may have just deleted, so they MUST
# be cleared too (the glob used to omit them, leaving a dangling best_iter ref).
# skill_baseline_*/ is intentionally NOT deleted — it is the canonical-baseline
# snapshot, taken once and valid regardless of which iter we roll back to.
banner "── Top-level cleanup ──"
for f in "${OUT_DIR}"/skill_final_* "${OUT_DIR}"/skill_best_* \
         "${OUT_DIR}"/best_skill_info.json \
         "${OUT_DIR}"/skill_changes_*.diff "${OUT_DIR}/cost_total.json"; do
    [[ -e "$f" ]] && del "$f"
done

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
if [[ -n "$AFTER_GAP" ]]; then
    echo -e "${BOLD}${GREEN}Done.${RESET}  Pipeline rolled back to: after gap-agent of iter ${RESTORE_ITER} (= start of iter $((RESTORE_ITER+1)))."
    echo -e "Skills restored from skill_after_* snapshots (post-gap state)."
    echo -e "Re-run with ${BOLD}--resume${RESET} to continue from here (weak agent will run for iter $((RESTORE_ITER+1))), e.g.:"
    echo -e "  ${DIM}bash run_skill_transfer.sh --resume --out-dir \"${OUT_DIR}\" ...${RESET}"
elif [[ -n "$KEEP_SESSION" ]]; then
    echo -e "${BOLD}${GREEN}Done.${RESET}  Pipeline rolled back to: iter ${RESTORE_ITER} gap-agent step (checkpoint resume)."
    echo -e "Session DB preserved — next run will resume from checkpoint automatically."
    echo -e "Re-run the pipeline or gap agent directly, e.g.:"
    echo -e "  ${DIM}bash run_skill_transfer.sh --resume --out-dir \"${OUT_DIR}\" ...${RESET}"
    echo -e "  ${DIM}# or: bash run_gap_resume_test.sh${RESET}"
else
    echo -e "${BOLD}${GREEN}Done.${RESET}  Pipeline rolled back to: iter ${RESTORE_ITER} gap-agent step (fresh start)."
    echo -e "Skills restored from skill_before_* snapshots — gap agent will restart from scratch."
    echo -e "Re-run with ${BOLD}--resume${RESET} to continue from here, e.g.:"
    echo -e "  ${DIM}bash run_skill_transfer.sh --resume --out-dir \"${OUT_DIR}\" ...${RESET}"
fi
