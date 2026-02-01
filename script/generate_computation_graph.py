#!/usr/bin/env python3
"""
Generate Transformer computation graph using Graphviz.

This script creates a visual representation of the forward pass through
a single TransformerLayer, showing how data flows from input indices
through embeddings, attention, feed-forward layers, to output logits.
"""

from graphviz import Digraph
import os


def create_computation_graph():
    """Create the Transformer computation graph."""

    dot = Digraph(comment='Transformer Computation Graph')
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # Node style definitions (from plan)
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

    # === Input Node ===
    dot.node('indices', 'indices', **input_style)

    # === Embedding Parameters ===
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('E_char', 'E_char', **param_style)
        s.node('E_pos', 'E_pos', **param_style)

    # === Embedding Outputs ===
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('e_c', 'e_c', **intermediate_style)
        s.node('e_p', 'e_p', **intermediate_style)

    # === Combined Embedding ===
    dot.node('x', 'x', **intermediate_style)

    # === Attention Parameters ===
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('W_q', 'W_q', **param_style)
        s.node('W_k', 'W_k', **param_style)
        s.node('W_v', 'W_v', **param_style)

    # === Q, K, V ===
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Q', 'Q', **intermediate_style)
        s.node('K', 'K', **intermediate_style)
        s.node('V', 'V', **intermediate_style)

    # === Attention Computation ===
    dot.node('scores', 'scores', **intermediate_style)
    dot.node('alpha', 'α', **intermediate_style)
    dot.node('attn', 'attn', **intermediate_style)

    # === Output Projection ===
    dot.node('W_o', 'W_o', **param_style)
    dot.node('a', 'a', **intermediate_style)

    # === First Residual ===
    dot.node('z', 'z', **intermediate_style)

    # === Feed-Forward Parameters ===
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('W_ff1', 'W_ff1', **param_style)
        s.node('W_ff2', 'W_ff2', **param_style)

    # === Feed-Forward Intermediates ===
    dot.node('h', 'h', **intermediate_style)
    dot.node('f', 'f', **intermediate_style)

    # === Second Residual ===
    dot.node('out', 'out', **intermediate_style)

    # === Output Layer ===
    dot.node('W_out', 'W_out', **param_style)
    dot.node('logits', 'logits', **intermediate_style)

    # === Final Output ===
    dot.node('y', 'y', **output_style)

    # === Edges ===

    # Embedding lookups
    dot.edge('indices', 'e_c')
    dot.edge('E_char', 'e_c', label='lookup')
    dot.edge('E_pos', 'e_p', label='lookup')

    # Combine embeddings
    dot.edge('e_c', 'x', label='+')
    dot.edge('e_p', 'x')

    # Q, K, V projections
    dot.edge('x', 'Q')
    dot.edge('x', 'K')
    dot.edge('x', 'V')
    dot.edge('W_q', 'Q', label='xW')
    dot.edge('W_k', 'K', label='xW')
    dot.edge('W_v', 'V', label='xW')

    # Attention scores
    dot.edge('Q', 'scores', label='QKᵀ/√d')
    dot.edge('K', 'scores')

    # Softmax
    dot.edge('scores', 'alpha', label='softmax')

    # Attention output
    dot.edge('alpha', 'attn', label='αV')
    dot.edge('V', 'attn')

    # Output projection
    dot.edge('attn', 'a')
    dot.edge('W_o', 'a', label='xW')

    # First residual connection
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


def main():
    """Generate and save the computation graph."""

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    media_dir = os.path.join(project_dir, 'media')

    # Ensure media directory exists
    os.makedirs(media_dir, exist_ok=True)

    # Create the graph
    dot = create_computation_graph()

    # Output paths (without extension - graphviz adds it)
    base_path = os.path.join(media_dir, 'part0_computation_graph_graphviz')

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
