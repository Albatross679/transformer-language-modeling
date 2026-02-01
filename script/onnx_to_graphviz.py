#!/usr/bin/env python3
"""
Generate Transformer computation graph from ONNX model using Graphviz.

This script parses the ONNX model and creates a visual representation
that matches the style of the manually-created computation graph.
"""

import os
import onnx
from graphviz import Digraph


# Mapping from ONNX node names to friendly labels
# These match the node names in the original generate_computation_graph.py
NODE_LABEL_MAP = {
    '/model/embedding/Gather': ('e_c', 'E_char lookup'),
    '/model/positional_encoding/Add': ('x', 'e_c + e_p'),
    '/model/layers.0/W_q/Gemm': ('Q', 'xW_q'),
    '/model/layers.0/W_k/Gemm': ('K', 'xW_k'),
    '/model/layers.0/W_v/Gemm': ('V', 'xW_v'),
    '/model/layers.0/Transpose': (None, 'transpose'),  # Skip - merged into scores
    '/model/layers.0/MatMul': (None, 'QKᵀ'),  # Skip - merged into scores
    '/model/layers.0/Constant': (None, '√d'),  # Skip - merged into scores
    '/model/layers.0/Div': ('scores', 'QKᵀ/√d'),
    '/model/layers.0/Softmax': ('alpha', 'softmax'),
    '/model/layers.0/MatMul_1': ('attn', 'αV'),
    '/model/layers.0/W_o/Gemm': ('a', 'xW_o'),
    '/model/layers.0/Add': ('z', 'x + a'),
    '/model/layers.0/ff1/Gemm': (None, 'zW_ff1'),  # Skip - merged with ReLU into h
    '/model/layers.0/Relu': ('h', 'ReLU'),
    '/model/layers.0/ff2/Gemm': ('f', 'hW_ff2'),
    '/model/layers.0/Add_1': ('out', 'z + f'),
    '/model/output_proj/Gemm': ('logits', 'xW_out'),
    '/model/LogSoftmax': ('y', 'log_softmax'),
}

# Categorization for styling (matches original graph)
PARAM_NODES = {'E_char', 'E_pos', 'W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'W_ff2', 'W_out'}
INPUT_NODES = {'indices'}
OUTPUT_NODES = {'y'}


def get_node_style(node_id):
    """Return the appropriate style dict for a node."""
    param_style = {
        'shape': 'doublecircle',
        'style': 'filled',
        'fillcolor': 'white',
        'width': '0.6',
        'height': '0.6',
    }

    intermediate_style = {
        'shape': 'circle',
        'style': 'filled',
        'fillcolor': '#e8f4e8',
        'width': '0.5',
        'height': '0.5',
    }

    input_style = {
        'shape': 'box',
        'style': 'filled,rounded',
        'fillcolor': '#e8e8f4',
    }

    output_style = {
        'shape': 'box',
        'style': 'filled,rounded',
        'fillcolor': '#f4e8e8',
    }

    if node_id in INPUT_NODES:
        return input_style
    elif node_id in OUTPUT_NODES:
        return output_style
    elif node_id in PARAM_NODES:
        return param_style
    else:
        return intermediate_style


def create_onnx_computation_graph(onnx_path):
    """Create the Transformer computation graph from ONNX model."""

    # Load ONNX model
    model = onnx.load(onnx_path)
    graph = model.graph

    dot = Digraph(comment='Transformer Computation Graph (from ONNX)')
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # Track which nodes we've created
    created_nodes = set()

    # Build a mapping from output tensor names to ONNX nodes
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # === Create Input Node ===
    dot.node('indices', 'indices', **get_node_style('indices'))
    created_nodes.add('indices')

    # === Create Parameter Nodes ===
    # These come from the graph initializers (weights/biases)
    dot.node('E_char', 'E_char', **get_node_style('E_char'))
    dot.node('E_pos', 'E_pos', **get_node_style('E_pos'))
    created_nodes.add('E_char')
    created_nodes.add('E_pos')

    # Group embedding params together
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('E_char', 'E_char')
        s.node('E_pos', 'E_pos')

    # Add remaining parameter nodes as referenced by ONNX ops
    param_nodes_added = set()
    for node in graph.node:
        node_name = node.name
        if node_name in NODE_LABEL_MAP:
            node_id, label = NODE_LABEL_MAP[node_name]

            # Extract parameter name from certain node types
            if 'W_q' in node_name:
                dot.node('W_q', 'W_q', **get_node_style('W_q'))
                param_nodes_added.add('W_q')
            elif 'W_k' in node_name:
                dot.node('W_k', 'W_k', **get_node_style('W_k'))
                param_nodes_added.add('W_k')
            elif 'W_v' in node_name:
                dot.node('W_v', 'W_v', **get_node_style('W_v'))
                param_nodes_added.add('W_v')
            elif 'W_o' in node_name:
                dot.node('W_o', 'W_o', **get_node_style('W_o'))
                param_nodes_added.add('W_o')
            elif 'ff1' in node_name:
                dot.node('W_ff1', 'W_ff1', **get_node_style('W_ff1'))
                param_nodes_added.add('W_ff1')
            elif 'ff2' in node_name:
                dot.node('W_ff2', 'W_ff2', **get_node_style('W_ff2'))
                param_nodes_added.add('W_ff2')
            elif 'output_proj' in node_name:
                dot.node('W_out', 'W_out', **get_node_style('W_out'))
                param_nodes_added.add('W_out')

    created_nodes.update(param_nodes_added)

    # Group attention weight params together
    with dot.subgraph() as s:
        s.attr(rank='same')
        for p in ['W_q', 'W_k', 'W_v']:
            if p in param_nodes_added:
                s.node(p, p)

    # Group feed-forward weight params together
    with dot.subgraph() as s:
        s.attr(rank='same')
        for p in ['W_ff1', 'W_ff2']:
            if p in param_nodes_added:
                s.node(p, p)

    # === Create Intermediate and Output Nodes ===
    for node in graph.node:
        node_name = node.name
        if node_name in NODE_LABEL_MAP:
            node_id, label = NODE_LABEL_MAP[node_name]
            # Skip nodes marked as None (they are merged into other nodes)
            if node_id is not None and node_id not in created_nodes:
                dot.node(node_id, node_id, **get_node_style(node_id))
                created_nodes.add(node_id)

    # Add e_p node (positional embedding) - not in ONNX as separate node
    dot.node('e_p', 'e_p', **get_node_style('e_p'))
    created_nodes.add('e_p')

    # Group Q, K, V together
    with dot.subgraph() as s:
        s.attr(rank='same')
        for n in ['Q', 'K', 'V']:
            if n in created_nodes:
                s.node(n, n)

    # === Create Edges ===

    # Embedding lookups
    dot.edge('indices', 'e_c')
    dot.edge('E_char', 'e_c', label='lookup')
    dot.edge('E_pos', 'e_p', label='lookup')

    # Combine embeddings (the Add node in ONNX)
    dot.edge('e_c', 'x', label='+')
    dot.edge('e_p', 'x')

    # Q, K, V projections
    dot.edge('x', 'Q')
    dot.edge('x', 'K')
    dot.edge('x', 'V')
    dot.edge('W_q', 'Q', label='xW')
    dot.edge('W_k', 'K', label='xW')
    dot.edge('W_v', 'V', label='xW')

    # K transpose (optional - could skip showing this)
    # dot.edge('K', 'K_T', label='ᵀ')

    # Attention scores: Q @ K^T / sqrt(d)
    dot.edge('Q', 'scores', label='QKᵀ/√d')
    dot.edge('K', 'scores')

    # Softmax
    dot.edge('scores', 'alpha', label='softmax')

    # Attention output: alpha @ V
    dot.edge('alpha', 'attn', label='αV')
    dot.edge('V', 'attn')

    # Output projection
    dot.edge('attn', 'a')
    dot.edge('W_o', 'a', label='xW')

    # First residual connection (Add in ONNX)
    dot.edge('x', 'z', style='dashed', label='residual')
    dot.edge('a', 'z', label='+')

    # Feed-forward layer
    dot.edge('z', 'h')
    dot.edge('W_ff1', 'h', label='ReLU(zW)')
    dot.edge('h', 'f')
    dot.edge('W_ff2', 'f', label='xW')

    # Second residual connection
    dot.edge('z', 'out', style='dashed', label='residual')
    dot.edge('f', 'out', label='+')

    # Output layer
    dot.edge('out', 'logits')
    dot.edge('W_out', 'logits', label='xW')

    # Final softmax
    dot.edge('logits', 'y', label='log_softmax')

    return dot


def print_onnx_structure(onnx_path):
    """Print the ONNX graph structure for debugging."""
    model = onnx.load(onnx_path)
    graph = model.graph

    print("\n" + "=" * 70)
    print("ONNX Graph Structure")
    print("=" * 70)

    print("\nInputs:")
    for inp in graph.input:
        print(f"  {inp.name}")

    print("\nOutputs:")
    for out in graph.output:
        print(f"  {out.name}")

    print("\nNodes:")
    for i, node in enumerate(graph.node):
        print(f"  [{i:2d}] {node.op_type:15s}: {node.name}")
        print(f"       inputs: {list(node.input)}")
        print(f"       outputs: {list(node.output)}")

    print("=" * 70 + "\n")


def main():
    """Generate and save the computation graph from ONNX."""

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'model')
    media_dir = os.path.join(project_dir, 'media')

    # ONNX model path
    onnx_path = os.path.join(model_dir, 'transformer_part1.onnx')

    # Check if model exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        print("Please run the export script first to create the ONNX model.")
        return

    # Ensure media directory exists
    os.makedirs(media_dir, exist_ok=True)

    # Print ONNX structure for debugging
    print_onnx_structure(onnx_path)

    # Create the graph
    dot = create_onnx_computation_graph(onnx_path)

    # Output paths (without extension - graphviz adds it)
    base_path = os.path.join(media_dir, 'part0_computation_graph_onnx')

    # Render as SVG
    dot.render(base_path, format='svg', cleanup=True)
    print(f"Generated: {base_path}.svg")

    # Render as PNG
    dot.render(base_path, format='png', cleanup=True)
    print(f"Generated: {base_path}.png")

    # Also save the DOT source for reference
    dot_path = base_path + '.dot'
    with open(dot_path, 'w') as f:
        f.write(dot.source)
    print(f"Generated: {dot_path}")


if __name__ == '__main__':
    main()
