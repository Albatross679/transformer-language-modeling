#!/usr/bin/env python3
"""
Demo: Force-Directed Layout with Graphviz

Shows how to use the force-directed layout algorithms to position nodes,
then render with Graphviz using the 'neato' engine which respects positions.
"""

import os
from graphviz import Digraph
from force_directed_layout import ForceDirectedLayout, KamadaKawai, FruchtermanReingold


def create_mlp_graph():
    """Create the MLP computation graph (like the reference image)."""
    nodes = ['x', 'W', 'b', 'h', 'V', 'a', 'y']
    edges = [
        ('x', 'h'), ('W', 'h'), ('b', 'h'),  # h = tanh(Wx + b)
        ('h', 'y'), ('V', 'y'), ('a', 'y'),  # y = Vh + a
    ]
    return nodes, edges


def create_transformer_graph():
    """Create a simplified Transformer computation graph."""
    nodes = [
        # Input/Embedding
        'idx', 'W_e', 'PE', 'x0',
        # QKV
        'W_q', 'W_k', 'W_v', 'Q', 'K', 'V',
        # Attention
        'QK', 'A',
        # Output
        'AV', 'W_o', 'out',
    ]
    edges = [
        ('idx', 'x0'), ('W_e', 'x0'), ('PE', 'x0'),
        ('x0', 'Q'), ('W_q', 'Q'),
        ('x0', 'K'), ('W_k', 'K'),
        ('x0', 'V'), ('W_v', 'V'),
        ('Q', 'QK'), ('K', 'QK'),
        ('QK', 'A'),
        ('A', 'AV'), ('V', 'AV'),
        ('AV', 'out'), ('W_o', 'out'),
    ]
    return nodes, edges


def render_with_force_layout(
    nodes, edges,
    algorithm='kamada-kawai',
    output_name='force_layout_demo',
    width=600, height=400
):
    """
    Compute force-directed layout and render with Graphviz.

    Args:
        nodes: List of node names
        edges: List of (source, target) tuples
        algorithm: 'spring', 'fruchterman-reingold', or 'kamada-kawai'
        output_name: Output filename (without extension)
        width, height: Layout dimensions
    """
    # Choose algorithm
    if algorithm == 'spring':
        layout_algo = ForceDirectedLayout(
            k_attractive=0.05,
            k_repulsive=2000,
            ideal_length=80
        )
    elif algorithm == 'fruchterman-reingold':
        layout_algo = FruchtermanReingold(cooling_factor=0.95)
    else:  # kamada-kawai
        layout_algo = KamadaKawai()

    # Compute positions
    positions = layout_algo.layout(
        nodes, edges,
        iterations=200,
        width=width,
        height=height,
        seed=42
    )

    # Create Graphviz graph
    dot = Digraph(comment=f'Force-Directed Layout ({algorithm})')
    dot.attr(engine='neato')  # neato respects pos attribute
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='8')

    # Style definitions
    param_style = {
        'shape': 'doublecircle',
        'style': 'filled',
        'fillcolor': '#E8F4FD',
        'width': '0.4',
        'height': '0.4',
    }

    activation_style = {
        'shape': 'doublecircle',
        'style': 'filled',
        'fillcolor': '#FFF3E0',
        'width': '0.4',
        'height': '0.4',
    }

    # Classify nodes (simple heuristic: W_ prefix = parameter)
    for node in nodes:
        x, y = positions[node]
        # Scale to inches (Graphviz default unit)
        pos_str = f"{x/72:.2f},{y/72:.2f}!"

        if node.startswith('W') or node in ['PE', 'a', 'b']:
            dot.node(node, node, pos=pos_str, **param_style)
        else:
            dot.node(node, node, pos=pos_str, **activation_style)

    # Add edges
    for src, tgt in edges:
        dot.edge(src, tgt)

    # Render
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    media_dir = os.path.join(project_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)

    output_path = os.path.join(media_dir, output_name)

    dot.render(output_path, format='png', cleanup=True)
    print(f"Saved: {output_path}.png")

    return positions


def compare_algorithms():
    """Compare all three algorithms on the same graph."""
    nodes, edges = create_mlp_graph()

    algorithms = ['spring', 'fruchterman-reingold', 'kamada-kawai']

    for algo in algorithms:
        print(f"\nRendering with {algo}...")
        positions = render_with_force_layout(
            nodes, edges,
            algorithm=algo,
            output_name=f'force_layout_{algo.replace("-", "_")}',
            width=400, height=300
        )

        print(f"  Node positions:")
        for node, (x, y) in sorted(positions.items()):
            print(f"    {node}: ({x:.1f}, {y:.1f})")


def main():
    print("Force-Directed Layout Demo")
    print("=" * 50)

    # Demo 1: Simple MLP graph
    print("\n1. MLP Computation Graph (Kamada-Kawai)")
    nodes, edges = create_mlp_graph()
    render_with_force_layout(
        nodes, edges,
        algorithm='kamada-kawai',
        output_name='force_layout_mlp'
    )

    # Demo 2: Transformer graph
    print("\n2. Transformer Computation Graph (Kamada-Kawai)")
    nodes, edges = create_transformer_graph()
    render_with_force_layout(
        nodes, edges,
        algorithm='kamada-kawai',
        output_name='force_layout_transformer'
    )

    # Demo 3: Compare algorithms
    print("\n3. Algorithm Comparison")
    compare_algorithms()

    print("\n" + "=" * 50)
    print("Done! Check the media/ directory for output images.")


if __name__ == '__main__':
    main()
