# Requirement Document Style

Generate hierarchical, checklist-style documents using the requirement document format.

## Trigger

Use when the user asks to:
- Create any structured document (plans, checklists, guides, specs)
- Create tasks or task lists for specific work
- Convert prose instructions into actionable items
- Organize requirements from a document/assignment/specification
- Break down a project or goal into trackable steps

## Input

The user provides:
- Source documents (paths, URLs, or pasted content), OR
- A description of tasks/goals to organize, OR
- A topic or project to create a structured plan for

## Output Format

### Frontmatter (YAML)

```yaml
---
id: <uuid>
name: <Document Title>
description: <One-line summary of what this requirements doc covers>
type: reference
created: <ISO 8601 timestamp>
updated: <ISO 8601 timestamp>
tags: [<relevant>, <tags>]
aliases: [<short-names>, <alternate-refs>]
---
```

### Document Structure

1. **Title**: `# <Document Name>` followed by key metadata (due date, course, etc.)

2. **Major Sections**: Use `## N. Section Name` with horizontal rule (`---`) before each

3. **Subsections**: Nest with decimal notation:
   - `### N.M Subsection`
   - `#### N.M.P Sub-subsection`

4. **Context Blocks**: Start sections with blockquote containing:
   ```markdown
   > Command: `<command to run>`
   > Target: <success metric>
   ```

5. **Checkboxes**: All actionable items use `- [ ]` format

6. **Item Style**:
   - Imperative fragments, not full sentences
   - Include technical details in parentheses: `Query projection (Linear)`
   - Use numbered lists (`1. [ ]`) for sequential steps
   - Use bullet lists (`- [ ]`) for unordered requirements

7. **Emphasis**:
   - `**NO**` or `**CRITICAL**` for prohibitions/warnings
   - `**Important**` for key notes
   - Inline `code` for commands, classes, files, arguments

8. **Cross-References**: Use `(see N.M)` to link related sections

9. **Warnings**: Use blockquote with bold prefix:
   ```markdown
   > **Warning**: <warning text>
   ```

## Writing Rules

1. **Extract, don't interpret**: Pull requirements directly from source; don't add your own
2. **Preserve technical terms**: Use exact class names, function names, file paths from source
3. **One checkbox = one verifiable action**: Each item should be completable and checkable
4. **Group by dependency**: Earlier sections should not depend on later ones
5. **Include success criteria**: Every major section needs a measurable target
6. **No prose paragraphs**: Convert explanations into structured items or omit
7. **Preserve all constraints**: Restrictions ("do NOT use X") get their own emphasized items

## Example Transformation

**Source prose:**
> "Your implementation should get over 95% accuracy. You must not use nn.TransformerEncoder or any off-the-shelf attention. Use Linear layers for Q, K, V projections."

**Output:**
```markdown
### 2.1 Implementation Requirements

#### 2.1.1 Accuracy Target
- [ ] Achieve > 95% accuracy on dev set

#### 2.1.2 Restrictions
- [ ] **NO** `nn.TransformerEncoder`
- [ ] **NO** off-the-shelf attention modules

#### 2.1.3 Attention Implementation
- [ ] Query projection (`nn.Linear`)
- [ ] Key projection (`nn.Linear`)
- [ ] Value projection (`nn.Linear`)
```

## Workflow

1. **Read** all source documents thoroughly
2. **Identify** major deliverables/parts → become top-level sections
3. **Extract** requirements for each part → become subsections
4. **Decompose** complex requirements → become sub-subsections with checkboxes
5. **Add** context blocks with commands and targets
6. **Add** cross-references between related sections
7. **Review** for completeness against source material
