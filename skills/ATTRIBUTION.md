# Third-party material in this directory

## `meta_skill_creator/skill-creator/`

Anthropic's `skill-creator` skill, redistributed unmodified under the **Apache
License 2.0** (Copyright 2026 Anthropic, PBC). The full licence text is at
`meta_skill_creator/skill-creator/LICENSE.txt`; the upstream source is
<https://github.com/anthropics/skills>.

This is the revision the experiments in the paper were run with, obtained via
the EvoSkill project. Upstream has since revised the skill; if you want the
current version, take it from the repository above — the pipeline reads
whatever is in this directory.

## `meta_brainstorming/brainstorming/`

Taken unmodified from the EvoSkill project
(<https://github.com/sentient-agi/EvoSkill>), distributed under the **Apache
License 2.0**; the licence text is at
<https://www.apache.org/licenses/LICENSE-2.0>.

## Not included

The domain skills the paper adapts are not redistributed here. Anthropic's
`xlsx` skill — the SpreadsheetBench baseline — carries a licence that
explicitly forbids reproduction, derivative works, and redistribution (the same
applies to `docx`, `pdf`, and `pptx`). Obtain it yourself from the repository
above. See `README.md` in this directory.
