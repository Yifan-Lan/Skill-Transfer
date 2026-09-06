#!/usr/bin/env bash
# setup_env.sh — configure the runtime environment for the Skill Transfer pipelines.
#
# SOURCE me (do not execute):
#     source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
#
# Pipeline entry scripts (run_skill_transfer.sh, run_eval.sh, etc.) source this
# at the top so every invocation lands in the same conda env (Python 3.11 +
# LibreOffice + all project deps) regardless of which shell the user starts from.
#
# Behaviour:
#   1. Locate conda (PATH first; fall back to ${SD_CONDA_ROOT}).
#   2. Activate ${SD_ENV_NAME} unless already active.
#   3. Export PYTHON=python so the entry scripts' default (python3.11) is
#      overridden — that binary does not exist on the Linux server.
#   4. Ensure ${CONDA_PREFIX}/lib is on LD_LIBRARY_PATH so soffice can find
#      libXinerama / libcairo / etc.  This is normally done by the conda
#      activation hook, but we re-assert it defensively.
#   5. Print a one-line summary so the user can confirm the env at a glance.
#
# Environment-variable overrides (set BEFORE sourcing):
#   SD_CONDA_ROOT       conda installation root (default: $HOME/miniconda3)
#   SD_ENV_NAME         conda env to activate   (default: skill-transfer)
#   SD_SKIP_ENV_SETUP=1 bypass entirely (use on macOS or where env is preconfigured)
#   SD_QUIET=1          suppress the one-line summary
#
# Cross-platform: on machines where conda or the named env is absent (e.g. a
# macOS contributor's laptop), this script prints a single WARN and returns
# without failing — the entry script then runs against whatever python/soffice
# happen to be on PATH.

# Allow bypass.
if [[ "${SD_SKIP_ENV_SETUP:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

_sd_conda_root="${SD_CONDA_ROOT:-$HOME/miniconda3}"
_sd_env_name="${SD_ENV_NAME:-skill-transfer}"

# 1. Initialise conda shell integration.
#    `source conda.sh` alone is NOT sufficient in a non-interactive subshell — it
#    loads the `conda` function but leaves `CONDA_SHLVL` unset, so a later
#    `conda activate` errors with "Run 'conda init' before 'conda activate'" AND
#    returns exit code 0, defeating any `if !` guard.  `conda shell.bash hook`
#    emits a complete shell integration snippet that works in every context.
_sd_conda_bin=""
if command -v conda >/dev/null 2>&1; then
    _sd_conda_bin="$(command -v conda)"
elif [[ -x "${_sd_conda_root}/bin/conda" ]]; then
    _sd_conda_bin="${_sd_conda_root}/bin/conda"
else
    echo "[setup_env] WARN: conda not on PATH and not at ${_sd_conda_root}/bin/conda; skipping activation." >&2
    return 0 2>/dev/null || exit 0
fi
eval "$("${_sd_conda_bin}" shell.bash hook 2>/dev/null)" || {
    echo "[setup_env] WARN: 'conda shell.bash hook' failed; skipping activation." >&2
    return 0 2>/dev/null || exit 0
}

# 2. Activate (idempotent).
#    Important: do NOT wrap `conda activate` in command substitution `$(...)` —
#    that creates a subshell, and `conda activate` mutates the *current* shell's
#    env, so the activation would be lost on subshell exit.  Instead, redirect
#    stderr to a temp file and rely on CONDA_DEFAULT_ENV as the truth signal.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${_sd_env_name}" ]]; then
    _sd_err_file="$(mktemp -t setup_env.XXXXXX 2>/dev/null || echo "/tmp/setup_env.$$.err")"
    conda activate "${_sd_env_name}" 2>"${_sd_err_file}"
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${_sd_env_name}" ]]; then
        echo "[setup_env] WARN: conda env '${_sd_env_name}' activation failed." >&2
        [[ -s "${_sd_err_file}" ]] && sed 's/^/[setup_env]   /' "${_sd_err_file}" >&2
        echo "[setup_env]       continuing with current env (${CONDA_DEFAULT_ENV:-system})." >&2
        rm -f "${_sd_err_file}"
        unset _sd_err_file _sd_conda_bin
        return 0 2>/dev/null || exit 0
    fi
    rm -f "${_sd_err_file}"
    unset _sd_err_file
fi
unset _sd_conda_bin

# 3. Override entry-script defaults unless caller already set them.
#    Scripts read $PYTHON; a few helpers read $PYTHON_BIN — both are set.
export PYTHON="${PYTHON:-python}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# 4. Defensive: ensure conda env lib dir is on LD_LIBRARY_PATH for soffice.
if [[ -n "${CONDA_PREFIX:-}" && ":${LD_LIBRARY_PATH:-}:" != *":${CONDA_PREFIX}/lib:"* ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# 5. One-line summary (suppressible).
if [[ "${SD_QUIET:-0}" != "1" ]]; then
    _sd_py="$(python --version 2>&1)"
    _sd_so="$(soffice --version 2>&1 | head -1 || echo 'soffice not found')"
    echo "[setup_env] env=${CONDA_DEFAULT_ENV:-?} PYTHON=${PYTHON} | ${_sd_py} | ${_sd_so}"
    unset _sd_py _sd_so
fi

unset _sd_conda_root _sd_env_name
