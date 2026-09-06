# Skills

Skill Transfer adapts a skill; it does not ship one. This directory is where
you place the skill you want to work with, alongside the two meta-skills that
the pipeline's own agents use.

## The skill under adaptation

Put it here as `<dir>/<skill-name>/SKILL.md`, and point `--skills-dir` at
`<dir>`:

```
skills/
└── my_baseline/
    └── xlsx/
        └── SKILL.md
```

The paper adapts third-party skills that we do not redistribute — Anthropic's
`xlsx` skill is licensed in a way that forbids it (see ATTRIBUTION.md). Get it
from <https://github.com/anthropics/skills>, or use a skill of your own; the
pipeline does not care where a skill came from.

## Meta-skills

The two LLM agents each activate a meta-skill that shapes how they work: the
Diagnoser activates `brainstorming` before diagnosing, and the Patcher
activates `skill-creator` before editing. Both are included here and are used
by default — see ATTRIBUTION.md for their licences.
