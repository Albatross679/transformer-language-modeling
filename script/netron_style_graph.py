#!/usr/bin/env python3
"""
Generate Netron-style computation graph from ONNX model.

Creates a visually appealing graph matching Netron's style:
- Color-coded operation nodes by type
- Tensor shapes displayed inside nodes
- Edge labels showing tensor dimensions
"""

import os
import onnx
from graphviz import Digraph


# Netron-style color palette for different operation types
OP_COLORS = {
    'Gather': '#4CAF50',      # Green
    'Gemm': '#2196F3',        # Blue
    'Add': '#607D8B',         # Blue-Gray
    'MatMul': '#607D8B',      # Blue-Gray
    'Div': '#607D8B',         # Blue-Gray
    'Transpose': '#388E3C',   # Dark Green
    'Softmax': '#8D6E63',     # Brown
    'LogSoftmax': '#8D6E63',  # Brown
    'Relu': '#FF9800',        # Orange
    'Constant': '#9E9E9E',    # Gray
    'default': '#333333',     # Dark gray default
}

# Text colors for contrast
OP_TEXT_COLORS = {
    'Gather': 'white',
    'Gemm': 'white',
    'Add': 'white',
    'MatMul': 'white',
    'Div': 'white',
    'Transpose': 'white',
    'Softmax': 'white',
    'LogSoftmax': 'white',
    'Relu': 'black',
    'Constant': 'white',
    'default': 'white',
}


def get_op_color(op_type):
    """Get background color for an operation type."""
    return OP_COLORS.get(op_type, OP_COLORS['default'])


def get_op_text_color(op_type):
    """Get text color for an operation type."""
    return OP_TEXT_COLORS.get(op_type, OP_TEXT_COLORS['default'])


def format_shape(shape):
    """Convert shape list to Netron-style format: [27, 64] → '⟨27×64⟩'."""
    if not shape:
        return ''
    dims = [str(d) if isinstance(d, int) and d > 0 else '?' for d in shape]
    return '⟨' + '×'.join(dims) + '⟩'


def get_edge_label(shape):
    """Get edge label from tensor shape (usually the first dimension)."""
    if not shape:
        return ''
    first_dim = shape[0]
    if isinstance(first_dim, int) and first_dim > 0:
        return str(first_dim)
    return ''


def get_tensor_shapes(model):
    """
    Extract tensor shapes from ONNX model.
    Returns dict: tensor_name -> shape list
    """
    shapes = {}
    graph = model.graph

    # From initializers (weights)
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)

    # From graph inputs
    for inp in graph.input:
        if inp.type.HasField('tensor_type'):
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            shapes[inp.name] = shape

    # From graph outputs
    for out in graph.output:
        if out.type.HasField('tensor_type'):
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            shapes[out.name] = shape

    # From value_info (intermediate tensors)
    for vi in graph.value_info:
        if vi.type.HasField('tensor_type'):
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            shapes[vi.name] = shape

    return shapes


def infer_output_shapes(model, shapes):
    """
    Infer output shapes for nodes where ONNX doesn't provide them.
    This is a simplified inference for common operations.
    """
    graph = model.graph

    for node in graph.node:
        output_name = node.output[0] if node.output else None
        if not output_name or output_name in shapes:
            continue

        op = node.op_type

        if op == 'Gather':
            # Gather: output shape depends on data and indices
            # For embedding lookup: [vocab_size, embed_dim] + [seq_len] -> [seq_len, embed_dim]
            if len(node.input) >= 2:
                data_shape = shapes.get(node.input[0], [])
                indices_shape = shapes.get(node.input[1], [])
                if data_shape and indices_shape:
                    # Axis 0 gather: [V, D] + [N] -> [N, D]
                    shapes[output_name] = indices_shape + data_shape[1:]

        elif op == 'Add':
            # Add: output shape is broadcasted shape (usually same as inputs)
            for inp in node.input:
                if inp in shapes:
                    shapes[output_name] = shapes[inp]
                    break

        elif op == 'Gemm':
            # Gemm: Y = alpha * A @ B + beta * C
            # Output shape: [batch, out_features] or just [out_features]
            if len(node.input) >= 2:
                a_shape = shapes.get(node.input[0], [])
                b_shape = shapes.get(node.input[1], [])
                if a_shape and b_shape:
                    # A: [M, K], B: [K, N] -> [M, N]
                    # Check for transB attribute
                    trans_b = False
                    for attr in node.attribute:
                        if attr.name == 'transB':
                            trans_b = attr.i == 1
                    if trans_b:
                        out_dim = b_shape[0]  # B is [N, K], output has N
                    else:
                        out_dim = b_shape[1] if len(b_shape) > 1 else b_shape[0]
                    if len(a_shape) > 1:
                        shapes[output_name] = [a_shape[0], out_dim]
                    else:
                        shapes[output_name] = [out_dim]

        elif op == 'Transpose':
            # Simple transpose
            if node.input and node.input[0] in shapes:
                inp_shape = shapes[node.input[0]]
                shapes[output_name] = inp_shape[::-1]

        elif op == 'MatMul':
            # MatMul: A @ B
            if len(node.input) >= 2:
                a_shape = shapes.get(node.input[0], [])
                b_shape = shapes.get(node.input[1], [])
                if a_shape and b_shape:
                    # [M, K] @ [K, N] -> [M, N]
                    out_shape = [a_shape[0]] if a_shape else []
                    if len(b_shape) > 1:
                        out_shape.append(b_shape[1])
                    elif b_shape:
                        out_shape.append(b_shape[0])
                    shapes[output_name] = out_shape

        elif op == 'Div':
            # Div: element-wise, output same shape as input
            if node.input and node.input[0] in shapes:
                shapes[output_name] = shapes[node.input[0]]

        elif op in ('Softmax', 'LogSoftmax', 'Relu'):
            # These preserve shape
            if node.input and node.input[0] in shapes:
                shapes[output_name] = shapes[node.input[0]]

        elif op == 'Constant':
            # Constant: shape from value attribute
            for attr in node.attribute:
                if attr.name == 'value':
                    if attr.t.dims:
                        shapes[output_name] = list(attr.t.dims)
                    else:
                        shapes[output_name] = [1]  # Scalar

    return shapes


def create_html_node_label(op_type, node_name, tensor_info_lines):
    """
    Create an HTML-like label for Graphviz that mimics Netron's style.

    Structure:
    ┌─────────────────┐
    │  Op Type        │  (colored header)
    ├─────────────────┤
    │ tensor info     │  (white body)
    │ tensor info     │
    └─────────────────┘
    """
    bg_color = get_op_color(op_type)
    text_color = get_op_text_color(op_type)

    # Build HTML table
    rows = []

    # Header row with op type
    rows.append(f'<TR><TD BGCOLOR="{bg_color}" ALIGN="CENTER">'
                f'<FONT COLOR="{text_color}"><B>{op_type}</B></FONT></TD></TR>')

    # Body rows with tensor info
    if tensor_info_lines:
        for line in tensor_info_lines:
            rows.append(f'<TR><TD BGCOLOR="white" ALIGN="LEFT">'
                        f'<FONT COLOR="#333333" POINT-SIZE="10">{line}</FONT></TD></TR>')
    else:
        # Add empty row for consistent sizing
        rows.append('<TR><TD BGCOLOR="white" HEIGHT="10"></TD></TR>')

    # Combine into table
    table = f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
{chr(10).join(rows)}
</TABLE>
>'''
    return table


def create_input_node_label(name, shape):
    """Create label for input nodes."""
    shape_str = format_shape(shape)
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BGCOLOR="#E3F2FD" ALIGN="CENTER"><FONT COLOR="#1565C0"><B>{name}</B></FONT></TD></TR>
<TR><TD BGCOLOR="white"><FONT POINT-SIZE="10">{shape_str}</FONT></TD></TR>
</TABLE>
>'''


def create_output_node_label(name, shape):
    """Create label for output nodes."""
    shape_str = format_shape(shape)
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
<TR><TD BGCOLOR="#FFEBEE" ALIGN="CENTER"><FONT COLOR="#C62828"><B>{name}</B></FONT></TD></TR>
<TR><TD BGCOLOR="white"><FONT POINT-SIZE="10">{shape_str}</FONT></TD></TR>
</TABLE>
>'''


def get_friendly_tensor_name(tensor_name):
    """Extract a friendly name from ONNX tensor name."""
    # Remove common prefixes and suffixes
    name = tensor_name
    if name.startswith('model.'):
        name = name[6:]
    if name.endswith('_output_0'):
        name = name[:-9]
    if name.startswith('/model/'):
        name = name[7:]
    # Clean up the name
    name = name.replace('/', '.')
    name = name.replace('layers.0.', 'L0.')
    return name


def create_netron_style_graph(onnx_path):
    """Create a Netron-style computation graph from ONNX model."""

    model = onnx.load(onnx_path)
    graph = model.graph

    # Get tensor shapes
    shapes = get_tensor_shapes(model)
    shapes = infer_output_shapes(model, shapes)

    # Create Graphviz digraph
    dot = Digraph(comment='Transformer Computation Graph (Netron Style)')
    dot.attr(rankdir='TB')
    dot.attr('graph', splines='polyline', nodesep='0.5', ranksep='0.6')
    dot.attr('node', shape='none', fontname='Segoe UI, Roboto, Arial', fontsize='11')
    dot.attr('edge', fontname='Segoe UI, Roboto, Arial', fontsize='9', color='#666666')

    # Track created nodes and build connectivity
    node_ids = {}  # ONNX node name -> graphviz node id
    output_to_node = {}  # tensor name -> graphviz node id that produces it

    # Create input nodes
    for inp in graph.input:
        # Skip initializers (weights) - they're shown differently
        is_initializer = any(init.name == inp.name for init in graph.initializer)
        if not is_initializer:
            node_id = f'input_{inp.name}'.replace('.', '_').replace('/', '_')
            shape = shapes.get(inp.name, [])
            label = create_input_node_label(inp.name, shape)
            dot.node(node_id, label)
            output_to_node[inp.name] = node_id

    # Create operation nodes
    for i, node in enumerate(graph.node):
        node_id = f'node_{i}'
        node_ids[node.name] = node_id

        # Collect tensor info for the node body
        tensor_lines = []

        # Show weight tensor shapes for Gather and Gemm
        if node.op_type == 'Gather':
            # Show embedding table shape
            data_name = node.input[0] if node.input else ''
            data_shape = shapes.get(data_name, [])
            if data_shape:
                friendly_name = get_friendly_tensor_name(data_name)
                tensor_lines.append(f'{friendly_name} {format_shape(data_shape)}')

        elif node.op_type == 'Gemm':
            # Show weight and bias shapes
            if len(node.input) >= 2:
                w_name = node.input[1]
                w_shape = shapes.get(w_name, [])
                if w_shape:
                    tensor_lines.append(f'W {format_shape(w_shape)}')
            if len(node.input) >= 3:
                b_name = node.input[2]
                b_shape = shapes.get(b_name, [])
                if b_shape:
                    tensor_lines.append(f'B {format_shape(b_shape)}')

        elif node.op_type == 'Add':
            # Show output shape
            out_name = node.output[0] if node.output else ''
            out_shape = shapes.get(out_name, [])
            if out_shape:
                tensor_lines.append(f'out {format_shape(out_shape)}')

        elif node.op_type == 'Constant':
            # Show constant value shape
            out_name = node.output[0] if node.output else ''
            out_shape = shapes.get(out_name, [])
            if out_shape:
                tensor_lines.append(f'√d_k')

        elif node.op_type in ('MatMul', 'Transpose', 'Div', 'Softmax', 'LogSoftmax', 'Relu'):
            # Show output shape
            out_name = node.output[0] if node.output else ''
            out_shape = shapes.get(out_name, [])
            if out_shape:
                tensor_lines.append(f'{format_shape(out_shape)}')

        # Create the node with HTML label
        label = create_html_node_label(node.op_type, node.name, tensor_lines)
        dot.node(node_id, label)

        # Map outputs to this node
        for out in node.output:
            output_to_node[out] = node_id

    # Create output node
    for out in graph.output:
        node_id = f'output_{out.name}'.replace('.', '_').replace('/', '_')
        shape = shapes.get(out.name, [])
        label = create_output_node_label(out.name, shape)
        dot.node(node_id, label)

    # Create edges
    for i, node in enumerate(graph.node):
        node_id = f'node_{i}'

        for inp in node.input:
            # Find source node
            if inp in output_to_node:
                src_id = output_to_node[inp]
                # Get edge label from tensor shape
                inp_shape = shapes.get(inp, [])
                edge_label = get_edge_label(inp_shape)
                dot.edge(src_id, node_id, label=edge_label if edge_label else '')

    # Connect last node to output
    if graph.node:
        last_node = graph.node[-1]
        last_node_id = node_ids.get(last_node.name)
        if last_node_id and graph.output:
            out_name = graph.output[0].name
            out_id = f'output_{out_name}'.replace('.', '_').replace('/', '_')
            out_shape = shapes.get(out_name, [])
            edge_label = get_edge_label(out_shape)
            dot.edge(last_node_id, out_id, label=edge_label if edge_label else '')

    return dot


def main():
    """Generate and save the Netron-style computation graph."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'model')
    media_dir = os.path.join(project_dir, 'media')

    onnx_path = os.path.join(model_dir, 'transformer_part1.onnx')

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        print("Please run the export script first to create the ONNX model.")
        return

    os.makedirs(media_dir, exist_ok=True)

    print(f"Loading ONNX model from {onnx_path}...")
    dot = create_netron_style_graph(onnx_path)

    base_path = os.path.join(media_dir, 'part1_netron_style')

    # Render as SVG
    dot.render(base_path, format='svg', cleanup=True)
    print(f"Generated: {base_path}.svg")

    # Render as PNG
    dot.render(base_path, format='png', cleanup=True)
    print(f"Generated: {base_path}.png")

    # Save DOT source
    dot_path = base_path + '.dot'
    with open(dot_path, 'w') as f:
        f.write(dot.source)
    print(f"Generated: {dot_path}")

    print("\nDone! Open the SVG or PNG file to view the Netron-style graph.")


if __name__ == '__main__':
    main()
