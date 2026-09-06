# Skill Transfer

Code for *Skill Transfer: Adapting Agent Skills from Strong to Weak Agents*.

A skill that works well for a frontier model often fails on a smaller one: the
strong agent silently supplies procedural steps the skill leaves implicit, and
the weak agent skips them. Skill Transfer rewrites the skill so the weak agent
can execute it, **without touching any model weights**.

Each iteration runs the weak agent on a small set of training tasks, abstracts
both agents' trajectories into typed execution structures, diffs them to find
the steps the weak agent is missing, and patches the skill. Two LLM agents do
the work: a **Diagnoser** (writes a gap report) and a **Patcher** (applies it).

---

## 1. Install

```bash
git clone https://github.com/Yifan-Lan/Skill-Transfer.git
cd Skill-Transfer

conda create -n skill-transfer python=3.11 -y
conda activate skill-transfer
pip install -r requirements.txt
```

Spreadsheet tasks additionally need LibreOffice on the `PATH` (used to
recalculate formulas before scoring):

```bash
conda install -c conda-forge libreoffice     # or your system package manager
soffice --version                            # should print a version
```

`setup_env.sh` activates the conda env for you when the entry scripts run. If
your conda lives somewhere unusual, point it there:

```bash
export SD_CONDA_ROOT=/path/to/miniconda3   # where conda lives
export SD_ENV_NAME=skill-transfer         # env to activate
export SD_SKIP_ENV_SETUP=1                # ...or bypass this entirely
```

## 2. Credentials

```bash
export OPENAI_API_KEY="sk-..."
```

Azure OpenAI works too — set the endpoint and the key is picked up
automatically:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
```

Nothing is read from a file; the code only ever reads these environment
variables.

## 3. Get a skill to adapt

Skill Transfer starts from an existing skill that already works for the strong
agent. The paper uses third-party skills that we do not redistribute — see the
citations in the paper for their sources, and drop them into `skills/`:

```
skills/
├── xlsx_baseline/xlsx/SKILL.md          # SpreadsheetBench
├── officeqa_baseline/{finance,ripgrep}/ # OfficeQA
├── dabench_baseline/{data-scientist,statistical-analysis}/
├── meta_brainstorming/     # shipped — the Diagnoser activates this
└── meta_skill_creator/     # shipped — the Patcher activates this
```

A skill directory is just a folder of `<name>/SKILL.md` (plus any scripts and
reference files the skill needs). Your own skill works fine.

The two *meta-skills* are included and always active — they shape how the
Diagnoser and Patcher themselves work, and are not the skill being adapted.
The domain skills are not redistributed here: Anthropic's `xlsx` skill, the
SpreadsheetBench baseline, is licensed in a way that forbids it. Get it from
[anthropics/skills](https://github.com/anthropics/skills). See
`skills/ATTRIBUTION.md` for the licence details.

## 4. Prepare a dataset

The pipeline reads a directory containing `dataset.json` plus the task inputs.
Scripts for the datasets in the paper are in `data/prepare/`. To use your own
tasks, write a `dataset.json` — a list of records:

```json
[
  {
    "id": "task_001",
    "type": "dabench",
    "task_path": "tables",
    "file_name": "sales.csv",
    "question": "What is the total revenue in Q3?",
    "constraints": "Report a number rounded to 2 decimals.",
    "format": "@revenue[value]",
    "golden_answers": [["revenue", "18342.50"]]
  }
]
```

`type` selects how the case is built and scored: `dabench` (CSV question →
`output.txt`), `officeqa` (document question → `output.txt`), or omit it for
SpreadsheetBench-style workbook editing (input `.xlsx` → `output.xlsx`).
Ground-truth fields are stripped from what the agent sees, so it cannot read
the answer.

Split your tasks into a small **training** set (the paper uses 20) and a
**validation** set (30), each its own directory with its own `dataset.json`.

Sanity-check the layout before spending any tokens:

```bash
python skill_transfer/pipeline_helpers.py prepare \
    --dataset-dir data/mytask/train_20 \
    --out-dir     /tmp/prepare_check
ls /tmp/prepare_check/cases/          # one directory per task
```

## 5. Adapt the skill

```bash
bash run_skill_transfer.sh \
    --dataset-dir data/mytask/train_20 \
    --val-dir     data/mytask/validation_30 \
    --out-dir     runs/my_first_run \
    --skills-dir  skills/xlsx_baseline \
    --strong-model gpt-5.4 \
    --weak-model   gpt-4.1-mini \
    --max-edits    6
```

`--max-edits` is the iteration budget. Start with `--max-edits 1` and a
handful of tasks to confirm everything is wired up before running the full
budget — one iteration of 20 tasks is the smallest useful unit of work.

The run is resumable. If it dies, re-run the same command with `--resume` and
it skips every step whose output already exists.

**Where the results land** — under `--out-dir`:

| Path | What it is |
| --- | --- |
| `skill_final_<name>/` | the skill after the last iteration |
| `skill_best_<name>/` | the iterate that scored best on the validation set |
| `skill_baseline_<name>/` | the unmodified skill you started from |
| `skill_changes_<name>.diff` | baseline → final, as a diff |
| `best_skill_info.json` | which iteration won, and its score |
| `cost_total.json` | token usage and cost |
| `batch_00/iter_NN/diagnoser/gap_report.json` | the gaps found that iteration |
| `batch_00/iter_NN/patcher/` | what the Patcher changed |

## 6. Evaluate

Score any skill on a held-out split:

```bash
bash run_eval.sh \
    --dataset-dir data/mytask/test_200 \
    --out-dir     runs/eval_adapted \
    --skills-dir  runs/my_first_run/skill_best_xlsx \
    --weak-model  gpt-4.1-mini
```

Results go to `runs/eval_adapted/eval.json`. For the comparisons in the paper,
run the same command three more times against `skill_baseline_xlsx`, and with
`--no-skill` to measure the agent with no skill at all.

`--pass-at-n N` runs each task N times and reports Pass@k for k = 1..N.

## 7. Troubleshooting

**"No skills were patched (no training cases found in any batch)"** — the strong
agent solved none of your training tasks, so there was nothing to learn from.
Scroll up for `[run-agents] <case> reason: ...` lines, which carry the real
cause. The two usual ones:

* `AuthenticationError: 401` — the key in `OPENAI_API_KEY` is wrong, or you meant
  to use Azure and did not set `AZURE_OPENAI_ENDPOINT`.
* the tasks are genuinely too hard for the strong model, or the scoring is
  rejecting correct answers. Check one case by hand:
  `cat <out-dir>/strong_eval.json` gives the verdict and the reason per task.

Skill Transfer only trains on tasks the *strong* agent solves, so a run with
zero solved tasks exits cleanly having done nothing — that is by design, not a
crash.

**`soffice: not found`** — only spreadsheet tasks need LibreOffice; install it
as in step 1, or use CSV/document tasks instead.

## 8. If a run goes wrong

Roll back to the start of an iteration and re-run from there:

```bash
bash restore_run.sh \
    --out-dir    runs/my_first_run \
    --skills-dir skills/xlsx_baseline \
    --iter       2 \
    --dry-run                 # drop --dry-run to actually do it
```

---

## Layout

```
run_skill_transfer.sh    adapt a skill          (main entry)
run_eval.sh              score a skill on a split
restore_run.sh           roll a run back to an earlier iteration
setup_env.sh             conda activation used by the above
skill_transfer/          the pipeline
skills/                  the skill you're adapting + the two meta-skills
data/prepare/            dataset builders for the benchmarks in the paper
```

Inside `skill_transfer/`, the pieces worth knowing about:

| File | Role |
| --- | --- |
| `skill_agent.py` | the agent under test: loads a skill, uses tools, produces output |
| `trajectory_abstractor.py` | trajectory → typed execution structure |
| `gap_diagnoser.py` | structures → ranked gap report |
| `skill_patcher.py` | gap report → edited skill |
| `validator_agent.py` | per-case comparison of weak output against strong |
| `pipeline_helpers.py` | prepare / run / evaluate, driven by the shell scripts |

## Benchmark repositories

`compute_cell_match.py` reuses SpreadsheetBench's own evaluator, and OfficeQA
scoring reuses that benchmark's `reward.py`. Clone those repositories next to
this one, or point at them explicitly:

```bash
export SPREADSHEETBENCH_DIR=/path/to/SpreadsheetBench
export OFFICEQA_DIR=/path/to/officeqa
```

## Citation

```bibtex
```
