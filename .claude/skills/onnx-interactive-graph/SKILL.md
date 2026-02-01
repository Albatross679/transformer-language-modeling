# ONNX Interactive Graph Viewer

Generate an interactive HTML viewer for ONNX computation graphs with draggable nodes and clean mathematical notation.

## Trigger

Use when the user asks to:
- Create an interactive graph viewer for an ONNX model
- Visualize a computation graph from ONNX
- Generate a draggable/editable graph visualization
- Create a computation graph viewer

## Input

The user provides:
- Path to an ONNX model file (`.onnx`), OR
- A model that can be exported to ONNX

## Output Format

Two files are generated:

1. **Python script**: `script/onnx_to_interactive_html.py`
2. **HTML output**: `media/<model_name>_interactive_graph.html`

## Visual Style

### 4-Color Scheme (Strict)

| Color | Hex | Node Type |
|-------|-----|-----------|
| Blue | `#2196F3` | Input / Output |
| Dark Gray | `#37474F` | Operations (Gemm, MatMul, Add, Softmax, etc.) |
| Purple | `#7B1FA2` | Learnable Parameters (weights, biases) |
| Orange | `#FF9800` | Constants (√dₖ, non-learnable values) |

### Node Structure

Each node displays:
1. **Header**: Formula with equals sign (e.g., `Q = X·Wq + b`)
2. **Body**: Output variable and shape (e.g., `Q (20,64)`)

```
┌─────────────────┐
│  Q = X·Wq + b   │  ← formula in header
├─────────────────┤
│  Q (20,64)      │  ← output variable and shape
└─────────────────┘
```

### Shape Format

Use tuple format: `(dim1,dim2)` not `⟨dim1×dim2⟩`
- Example: `(20,64)` for a 20×64 matrix
- Example: `(27,64)` for embedding table

### Separate Parameter Nodes

Each learnable parameter must have its own node:
- `E` - Embedding matrix
- `PE` - Positional encoding
- `Wq`, `Wk`, `Wv` - Attention projection weights
- `Wo` - Output projection weight
- `W1`, `W2` - FFN weights
- `b` - Biases

Do NOT list parameters as "inputs" inside operation nodes. Create separate purple nodes for each.

### Formula Examples

| Operation | Formula |
|-----------|---------|
| Embedding | `e = E[x]` |
| Linear | `Q = X·Wq + b` |
| MatMul | `scores = Q·Kᵀ` |
| Scale | `s = X / √d` |
| Softmax | `α = softmax(X)` |
| Add | `z = A + B` |
| ReLU | `h = ReLU(X)` |

## Navigation Controls

### Scroll = Pan (NOT Zoom)
- Scroll wheel / two-finger swipe pans the view
- NO zoom on scroll
- NO drag-to-pan on background (disabled)

### Zoom via Buttons Only
- `+` button to zoom in
- `−` button to zoom out
- `⌂` button to reset view

### Multi-Select and Bulk Drag
- **Shift+Click** or **Ctrl+Click** to add/remove nodes from selection
- **Drag any selected node** to move all selected nodes together
- **Click on empty space** to deselect all
- **Visual highlight** (red border) on all selected nodes

### Rectangle Selection (Marquee)
- **Click and drag on empty canvas** to draw a selection rectangle
- **All nodes overlapping the rectangle** are added to selection
- **Shift+drag** to add to existing selection (instead of replacing)
- Dashed blue rectangle shown while dragging

### Selection Panel
- Single selection: shows node ID, type, x, y coordinates (editable)
- Multi-selection: shows "N nodes selected", type shows "multiple"

## HTML Features

The generated HTML must include:

1. **SVG rendering**: Crisp vector graphics
2. **Draggable nodes**: Click and drag to reposition
3. **Auto-updating edges**: Curved bezier paths that follow nodes
4. **Export button**: Save positions as JSON
5. **Sidebar**: Legend (3 colors only), selected node info, instructions
6. **No external dependencies**: Works offline, self-contained

## Layout Rules

1. **Input nodes** (blue) are one layer **above** the operations they feed into
2. **Parameter nodes** (purple) are one layer **above** the operations they feed into
3. **Constant nodes** (orange) are one layer **above** the operations they feed into
4. **Operation nodes** (gray) flow top-to-bottom based on data dependencies
5. **Output nodes** (blue) are one layer **below** the last operation
6. All input/parameter nodes are **centered horizontally** above their child operations
7. Minimize edge lengths by positioning nodes near connected neighbors
8. **Center-align all layers** - Each layer (row) of nodes is centered horizontally around the canvas center, including single-node layers
9. **Connectivity-based ordering** - Within each layer, more connected nodes (more edges) are placed closer to the center, less connected nodes toward the edges

## Style Rules

1. Use only 4 colors for nodes (blue=I/O, gray=ops, purple=params, orange=constants)
2. Every weight/bias must have its own node
3. Shape format must be `(dim,dim)` tuple style
4. No "In:" or "Out:" prefixes in body text
5. Scroll pans, does NOT zoom
6. Formulas must have equals sign showing output variable
7. All styling and JavaScript inline (self-contained HTML)
8. Handle "onnx::" prefixed names by extracting the operation name

## Example

**Input:**
> Create an interactive graph viewer for model/transformer_part1.onnx

**Output:**
```bash
# Generate the HTML
python script/onnx_to_interactive_html.py

# Serve on port 8080
cd media && python -m http.server 8080

# Access at: http://localhost:8080/part1_interactive_graph.html
```

## Template Code Structure

See [template.py](template.py) for the Python script template.
