---
id: ce2a6ecc-179c-462a-b5c5-5439aa108be2
name: markdown
description: Use when creating, modifying, or organizing any markdown file. Ensures Dendron and Obsidian compatibility with proper frontmatter, naming conventions, and wikilinks. Dot-notation naming only for experiment, issues, and log types.
type: reference
created: 2026-01-29T20:53:54
updated: 2026-01-30T18:15:00
tags: [skill, markdown, dendron, obsidian]
aliases: []
---

# Dendron-Obsidian Compatible Markdown

## Frontmatter Template (Required)

Every note MUST include this frontmatter structure:

```yaml
---
id: <uuid-v4>
name: <note-name>
description: <brief description>
type: <document-type>
created: <ISO-datetime>
updated: <ISO-datetime>
tags: [tag1, tag2]
aliases: [alternate-name]
---
```

### Field Requirements

| Field | Required | Format | Notes |
|-------|----------|--------|-------|
| `id` | Yes | UUID v4 | Dendron requires; Obsidian ignores |
| `name` | Yes | String | Short name (matches SKILL.md format) |
| `description` | Yes | String | Brief description of the document |
| `type` | Yes | String | Document type (see below) |
| `created` | Yes | ISO datetime | `2025-01-29T14:30:00` (seconds precision) |
| `updated` | Yes | ISO datetime | Same format as created |
| `tags` | No | Array `[tag1, tag2]` | Both support |
| `aliases` | No | Array `[alt-name]` | Both support |

### Document Types

| Type | Purpose |
|------|---------|
| `plan` | Intent, goals, roadmaps |
| `experiment` | Hypothesis testing, trials |
| `log` | Chronological records, decisions, results |
| `issues` | Problems, bugs, blockers |
| `notes` | General notes, meeting notes, research |
| `reference` | API docs, library usage, concepts |

Generate UUID with: `uuidgen` or Python `uuid.uuid4()`

## File Naming Convention

The project root serves as the vault. **Dot-notation filenames are ONLY used for `experiment`, `issues`, and `log` types.** All other types use simple hyphenated names.

```
project/                    # vault root
├── doc/                    # documentation
│   ├── project-alpha-plan.md          # plan (no dot notation)
│   ├── experiment.dan-hyperparams.md  # experiment (dot notation)
│   ├── issues.sigmoid-overflow.md     # issues (dot notation)
│   ├── log.2025-01-29.md              # log (dot notation)
│   ├── meeting-kickoff-notes.md       # notes (no dot notation)
│   └── glove-embeddings-reference.md  # reference (no dot notation)
├── .claude/
│   └── skills/
│       ├── markdown/
│       │   └── SKILL.md
│       └── latex/
│           └── SKILL.md
├── src/
├── data/
└── README.md
```

**Rules:**
- Project folder = vault root
- `CLAUDE.md` and `README.md` are exceptions (no frontmatter required)
- **Dot-notation types** (`experiment`, `issues`, `log`): `<type>.<topic>.md`
- **Simple naming types** (`plan`, `notes`, `reference`): `<topic>-<type>.md` or `<descriptive-name>.md`
- Lowercase, hyphenated words
- No spaces in filenames

## Linking Syntax

**Use only basic wikilinks:**

```markdown
[[note-name]]
```

**Avoid these (compatibility issues):**

| Syntax | Problem |
|--------|---------|
| `[[note\|display text]]` | Inconsistent pipe handling |
| `[[note#heading]]` | Heading links differ |
| `[[note#^block-id]]` | Obsidian-only |
| `![[embed]]` | Rendering differs |

**Safe alternatives:**
- Link then explain: `[[note-name]] (see section on X)`
- Use standard markdown links for external: `[text](url)`

### Callouts (Limited Compatibility)
Obsidian callouts don't render in Dendron. Use blockquotes instead:

```markdown
> **Note:** Important information here.
```

## Features to Avoid

| Feature | Reason |
|---------|--------|
| Dataview queries | Obsidian-only |
| Dendron schemas | Dendron-only |
| Mermaid diagrams | Inconsistent rendering |
| `<%` Templater syntax | Obsidian-only |
| Dendron note refs `![[ref]]` | Different behavior |
| Obsidian comments `%%` | Obsidian-only |

## Example Notes

### Example 1: Plan (simple naming)

Filename: `doc/project-alpha-plan.md`

```markdown
---
id: 8a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
name: Project Alpha Plan
description: Planning document for Project Alpha including goals and timeline.
type: plan
created: 2025-01-29T10:30:00
updated: 2025-01-29T10:30:00
tags: [project, active]
aliases: [alpha-plan]
---

## Overview

This document describes the plan for Project Alpha.

## Goals

- [x] Initial planning
- [ ] Development phase

## Notes

> **Important:** Deadline is end of Q1.

See [[issues.project-alpha]] for known blockers.
```

### Example 2: Experiment (dot notation)

Filename: `doc/experiment.learning-rate-sweep.md`

```markdown
---
id: 1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
name: Learning Rate Sweep
description: Experiment testing different learning rates for model training.
type: experiment
created: 2025-01-29T14:00:00
updated: 2025-01-29T14:00:00
tags: [experiment, hyperparameter]
aliases: [lr-sweep]
---

## Hypothesis

Lower learning rates will improve convergence stability.

## Results

...
```

## Generating Compliant Notes

When creating new markdown files:

1. Generate UUID: `python -c "import uuid; print(uuid.uuid4())"`
2. Get ISO datetime: `date -Iseconds` or `date +%Y-%m-%dT%H:%M:%S`
3. Apply frontmatter template
4. Choose filename based on type:
   - `experiment`, `issues`, `log` → use dot notation: `<type>.<topic>.md`
   - `plan`, `notes`, `reference` → use simple naming: `<descriptive-name>.md`
5. Link with basic `[[wikilinks]]` only
