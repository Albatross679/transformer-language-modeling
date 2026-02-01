# Create Claude Code Skill

Create custom skills for Claude Code following the standard skill structure.

## Trigger

Use when the user asks to:
- Create a new Claude Code skill
- Add a custom skill to the project
- Set up a skill for a specific workflow or task
- Document a repeatable process as a skill

## Required Structure

Each skill must be a **folder** with `SKILL.md` as the entry point:

```
.claude/skills/
└── skill-name/
    └── SKILL.md           # Required - main instructions
```

**Important**: Flat files like `.claude/skills/skill-name.md` are NOT recognized.

## Optional Supporting Files

Skills can include additional resources:

```
skill-name/
├── SKILL.md               # Required entry point
├── template.md            # Template for Claude to fill in
├── examples/              # Example outputs
│   └── sample.md
├── scripts/               # Scripts Claude can execute
│   └── validate.sh
└── reference/             # Reference documentation
    └── spec.md
```

### Reference File Restrictions

**All supporting files must be within the skill's directory.** You cannot reference files outside the skill folder.

| Location | Supported |
|----------|-----------|
| Within skill directory | ✓ `template.md` |
| Subdirectories | ✓ `examples/sample.md` |
| Outside skill folder | ✗ `../../data/file.txt` |
| Project root | ✗ `../../../README.md` |
| Absolute paths | ✗ `/home/user/file.txt` |

**How to reference supporting files in SKILL.md:**

```markdown
## Additional Resources

- For templates, see [template.md](template.md)
- For examples, see [examples/sample.md](examples/sample.md)
```

If you need external data, you must either:
1. Copy the files into the skill directory
2. Hardcode external paths in bash commands within the skill instructions

## SKILL.md Structure

### 1. Title and Description

```markdown
# Skill Name

Brief one-line description of what this skill does.
```

### 2. Trigger Section

Define when Claude should use this skill:

```markdown
## Trigger

Use when the user asks to:
- [Action verb] [specific task]
- [Action verb] [related task]
- [Action verb] [variant task]
```

### 3. Input Section (Optional)

Describe what input the skill expects:

```markdown
## Input

The user provides:
- [Required input type/format]
- [Optional input], OR
- [Alternative input]
```

### 4. Output Section

Specify the expected output format:

```markdown
## Output Format

[Describe structure, formatting rules, examples]
```

### 5. Workflow/Steps Section

Document the process:

```markdown
## Workflow

1. **Step name**: Description
2. **Step name**: Description
3. **Step name**: Description
```

### 6. Rules/Guidelines Section (Optional)

Add constraints or best practices:

```markdown
## Rules

1. [Constraint or guideline]
2. [Constraint or guideline]
```

### 7. Examples Section (Optional)

Show input/output examples:

```markdown
## Example

**Input:**
> [example input]

**Output:**
[example output]
```

## Naming Conventions

- Use lowercase with hyphens: `my-skill-name`
- Be descriptive but concise
- Avoid generic names like `helper` or `utility`

Good: `create-skill`, `latex`, `requirement-document-style`
Bad: `skill1`, `helper`, `misc`

## Best Practices

1. **Single responsibility**: Each skill should do one thing well
2. **Clear triggers**: Make it obvious when the skill applies
3. **Concrete examples**: Show expected input/output
4. **Actionable steps**: Use imperative verbs
5. **No ambiguity**: Be specific about formats and constraints
6. **Reference files**: Use relative paths to supporting files

## Template

```markdown
# [Skill Name]

[One-line description of the skill's purpose.]

## Trigger

Use when the user asks to:
- [Primary use case]
- [Secondary use case]
- [Variant use case]

## Input

The user provides:
- [Input description]

## Output Format

[Describe the expected output structure and format]

## Workflow

1. **[Step]**: [Description]
2. **[Step]**: [Description]
3. **[Step]**: [Description]

## Rules

1. [Key constraint]
2. [Key constraint]

## Example

**Input:**
> [Example input]

**Output:**
[Example output]
```

## Skill Locations

Skills can be placed in:
- **Project-level**: `.claude/skills/` (applies to current project)
- **User-level**: `~/.claude/skills/` (applies to all projects)

Project-level skills take precedence over user-level skills with the same name.
