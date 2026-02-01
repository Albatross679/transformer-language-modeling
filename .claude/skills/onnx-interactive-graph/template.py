#!/usr/bin/env python3
"""
Template for ONNX to Interactive HTML Graph Viewer.

This template shows the structure for generating interactive computation graph viewers.
"""

import os
import json
import onnx

# =============================================================================
# 4-COLOR PALETTE
# =============================================================================

COLORS = {
    'io': '#2196F3',           # Blue - Input/Output
    'op': '#37474F',           # Dark Gray - Operations
    'param': '#7B1FA2',        # Purple - Learnable Parameters
    'const': '#FF9800',        # Orange - Constants
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_shape(shape):
    """Convert shape list to tuple format: [27, 64] -> '(27,64)'."""
    if not shape:
        return ''
    dims = [str(d) if isinstance(d, int) and d > 0 else '?' for d in shape]
    return '(' + ','.join(dims) + ')'


def get_friendly_tensor_name(tensor_name):
    """Extract a friendly variable name from ONNX tensor name."""
    name = tensor_name.lower()

    # Map common patterns to short variable names
    if 'embedding' in name and 'weight' in name:
        return 'E'
    if 'positional' in name:
        return 'PE'
    if 'w_q' in name or '/q/' in name:
        return 'Wq'
    if 'w_k' in name or '/k/' in name:
        return 'Wk'
    if 'w_v' in name or '/v/' in name:
        return 'Wv'
    if 'w_o' in name:
        return 'Wo'
    if 'ff1' in name:
        return 'W1' if 'weight' in name else 'b1'
    if 'ff2' in name:
        return 'W2' if 'weight' in name else 'b2'
    if 'weight' in name:
        return 'W'
    if 'bias' in name:
        return 'b'

    return tensor_name[:8] if len(tensor_name) > 8 else tensor_name


def get_tensor_shapes(model):
    """Extract tensor shapes from ONNX model."""
    shapes = {}
    graph = model.graph

    for init in graph.initializer:
        shapes[init.name] = list(init.dims)

    for inp in graph.input:
        if inp.type.HasField('tensor_type'):
            shape = [d.dim_value if d.dim_value > 0 else '?'
                     for d in inp.type.tensor_type.shape.dim]
            shapes[inp.name] = shape

    return shapes


def parse_onnx_model(onnx_path):
    """
    Parse ONNX model and extract nodes, edges, and shapes.

    Key principles:
    1. Create separate nodes for all learnable parameters
    2. Use 3-color scheme: io, op, param
    3. Formulas with equals sign showing output variable
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    shapes = get_tensor_shapes(model)

    nodes_data = []
    edges_data = []
    output_to_node = {}
    initializer_names = {init.name for init in graph.initializer}

    node_idx = 0

    # 1. Create INPUT nodes (blue)
    for inp in graph.input:
        if inp.name not in initializer_names:
            node_id = f'input_{node_idx}'
            shape = shapes.get(inp.name, [])
            nodes_data.append({
                'id': node_id,
                'type': 'Input',
                'formula': 'x = input',
                'color': COLORS['io'],
                'text_color': 'white',
                'body_lines': [f'x {format_shape(shape)}'],
            })
            output_to_node[inp.name] = node_id
            node_idx += 1

    # 2. Create PARAMETER nodes (purple) - one for each weight/bias
    for node in graph.node:
        for inp_name in node.input:
            if inp_name in initializer_names and inp_name not in output_to_node:
                node_id = f'param_{node_idx}'
                shape = shapes.get(inp_name, [])
                var_name = get_friendly_tensor_name(inp_name)
                nodes_data.append({
                    'id': node_id,
                    'type': 'Param',
                    'formula': var_name,
                    'color': COLORS['param'],
                    'text_color': 'white',
                    'body_lines': [format_shape(shape)],
                })
                output_to_node[inp_name] = node_id
                node_idx += 1

    # 3. Create OPERATION nodes (dark gray)
    for node in graph.node:
        node_id = f'op_{node_idx}'
        op_type = node.op_type
        out_tensor = node.output[0] if node.output else ''
        out_shape = shapes.get(out_tensor, [])

        # Build formula based on operation type
        # Formula should be: output_var = operation(inputs)
        if op_type == 'Gather':
            formula = 'e = E[x]'
        elif op_type == 'Gemm':
            formula = 'y = X·W + b'
        elif op_type == 'MatMul':
            formula = 'y = A·B'
        elif op_type == 'Add':
            formula = 'z = A + B'
        elif op_type == 'Softmax':
            formula = 'α = softmax(X)'
        elif op_type == 'Div':
            formula = 's = X / √d'
        elif op_type == 'Transpose':
            formula = 'Kᵀ = Xᵀ'
        elif op_type == 'Relu':
            formula = 'h = ReLU(X)'
        else:
            formula = f'y = {op_type}(X)'

        nodes_data.append({
            'id': node_id,
            'type': 'Op',
            'formula': formula,
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': [f'y {format_shape(out_shape)}'],
        })

        # Map outputs and create edges
        for out in node.output:
            output_to_node[out] = node_id
        for inp_name in node.input:
            if inp_name in output_to_node:
                edges_data.append({
                    'source': output_to_node[inp_name],
                    'target': node_id,
                    'label': '',
                })
        node_idx += 1

    # 4. Create OUTPUT node (blue)
    for out in graph.output:
        node_id = f'output_{node_idx}'
        shape = shapes.get(out.name, [])
        nodes_data.append({
            'id': node_id,
            'type': 'Output',
            'formula': 'ŷ = output',
            'color': COLORS['io'],
            'text_color': 'white',
            'body_lines': [f'ŷ {format_shape(shape)}'],
        })
        # Connect last op to output
        if graph.node:
            last_out = graph.node[-1].output[0]
            if last_out in output_to_node:
                edges_data.append({
                    'source': output_to_node[last_out],
                    'target': node_id,
                    'label': '',
                })

    return nodes_data, edges_data


# =============================================================================
# HTML TEMPLATE (Key sections)
# =============================================================================

"""
Key JavaScript features:

1. SCROLL = PAN (not zoom):
   container.addEventListener('wheel', (e) => {
       e.preventDefault();
       panX -= e.deltaX;
       panY -= e.deltaY;
       updateTransform();
   });

2. ZOOM via buttons only:
   function zoomIn() { scale = Math.min(scale * 1.2, 5); updateTransform(); }
   function zoomOut() { scale = Math.max(scale * 0.8, 0.1); updateTransform(); }

3. Node dragging accounts for zoom scale:
   const dx = (e.clientX - startX) / scale;
   const dy = (e.clientY - startY) / scale;

4. NO drag-to-pan on background (only scroll/swipe pans)

5. MULTI-SELECT and BULK DRAG:
   - Shift+Click or Ctrl+Click to add/remove nodes from selection
   - Drag any selected node to move all selected nodes together
   - Click on empty space to deselect all
   - Red border highlight on all selected nodes
   - Selection state stored in: let selectedNodes = new Set();

6. RECTANGLE SELECTION (Marquee):
   - Click and drag on empty canvas to draw selection rectangle
   - All nodes overlapping the rectangle are added to selection
   - Shift+drag to add to existing selection
   - Selection state: isSelecting, selectStartX, selectStartY, selectionRect

7. Legend shows 4 colors:
   - Blue (#2196F3): Input/Output
   - Dark Gray (#37474F): Operations
   - Purple (#7B1FA2): Learnable Parameters
   - Orange (#FF9800): Constants

Layout rules:
- Input/Parameter/Constant nodes: ONE LAYER ABOVE the operations they feed into
- Centered horizontally above their child operations
- Operations flow top-to-bottom based on data dependencies
- Output nodes: one layer below the last operation
"""


def main():
    """Generate the interactive HTML viewer from ONNX model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    onnx_path = os.path.join(project_dir, 'model', 'transformer_part1.onnx')
    output_path = os.path.join(project_dir, 'media', 'part1_interactive_graph.html')

    nodes_data, edges_data = parse_onnx_model(onnx_path)
    # Generate positions using topological layout...
    # Generate HTML with embedded data...

    print(f"Generated: {output_path}")


if __name__ == '__main__':
    main()
