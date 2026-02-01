#!/usr/bin/env python3
"""
Generate Transformer computation graph using Kamada-Kawai force-directed layout.

The Kamada-Kawai algorithm (1989) minimizes stress so that Euclidean distances
approximate graph-theoretic distances:

    Stress = Σ w_ij × (||p_i - p_j|| - d_ij)²

    where:
      d_ij = shortest path length between nodes i and j
      w_ij = 1/d_ij² (weight inversely proportional to distance)

Benefits for computation graphs:
- Preserves graph structure naturally
- Connected nodes stay close, distant nodes spread out
- No manual rank/position constraints needed
- Handles complex edge patterns (residuals, fan-out) gracefully

Workflow:
1. Run script to generate initial layout → saves positions to JSON
2. Edit the JSON file to fine-tune positions manually
3. Re-run script → uses your edited positions instead of recomputing
"""

import os
import sys
import json
import argparse

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from force_directed_layout import KamadaKawai, layout_to_graphviz
from graphviz import Digraph


def define_transformer_graph():
    """
    Define the Transformer computation graph nodes and edges.

    Returns:
        nodes: List of node IDs
        edges: List of (source, target) tuples
        node_info: Dict mapping node ID to (label, category)
    """
    # Node categories: 'param', 'op', 'activation', 'constant'
    node_info = {
        # Input & Embedding
        'idx':      ('idx', 'activation'),
        'W_emb':    ('Wₑ', 'param'),
        'PE':       ('PE', 'param'),
        'emb':      ('emb', 'activation'),
        'x0':       ('x₀', 'activation'),

        # Q, K, V Projections
        'W_q':      ('Wq', 'param'),
        'b_q':      ('bq', 'param'),
        'Q':        ('Q', 'activation'),

        'W_k':      ('Wk', 'param'),
        'b_k':      ('bk', 'param'),
        'K':        ('K', 'activation'),

        'W_v':      ('Wv', 'param'),
        'b_v':      ('bv', 'param'),
        'V':        ('V', 'activation'),

        # Attention
        'sqrt_d':   ('√d', 'constant'),
        'scores':   ('s', 'activation'),
        'attn':     ('A', 'activation'),
        'attn_out': ('aₒ', 'activation'),

        # Output projection
        'W_o':      ('Wₒ', 'param'),
        'b_o':      ('bₒ', 'param'),
        'proj':     ('p', 'activation'),
        'x1':       ('x₁', 'activation'),

        # FFN
        'W_ff1':    ('Wf₁', 'param'),
        'b_ff1':    ('bf₁', 'param'),
        'h':        ('h', 'activation'),

        'W_ff2':    ('Wf₂', 'param'),
        'b_ff2':    ('bf₂', 'param'),
        'ff2':      ('f₂', 'activation'),
        'x2':       ('x₂', 'activation'),

        # Output layer
        'W_out':    ('Wᵤ', 'param'),
        'b_out':    ('bᵤ', 'param'),
        'logits':   ('l', 'activation'),
        'y_hat':    ('ŷ', 'activation'),

        # Operation nodes
        'op_lookup':  ('lookup', 'op'),
        'op_pe_add':  ('+', 'op'),
        'op_q':       ('lin', 'op'),
        'op_k':       ('lin', 'op'),
        'op_v':       ('lin', 'op'),
        'op_qk':      ('×', 'op'),
        'op_scale':   ('÷', 'op'),
        'op_softmax': ('σ', 'op'),
        'op_av':      ('×', 'op'),
        'op_o':       ('lin', 'op'),
        'op_res1':    ('+', 'op'),
        'op_ff1':     ('lin', 'op'),
        'op_relu':    ('relu', 'op'),
        'op_ff2':     ('lin', 'op'),
        'op_res2':    ('+', 'op'),
        'op_out':     ('lin', 'op'),
        'op_logsm':   ('logσ', 'op'),
    }

    nodes = list(node_info.keys())

    # Data flow edges (directed)
    edges = [
        # Input & Embedding
        ('idx', 'op_lookup'),
        ('W_emb', 'op_lookup'),
        ('op_lookup', 'emb'),
        ('emb', 'op_pe_add'),
        ('PE', 'op_pe_add'),
        ('op_pe_add', 'x0'),

        # Q projection
        ('x0', 'op_q'),
        ('W_q', 'op_q'),
        ('b_q', 'op_q'),
        ('op_q', 'Q'),

        # K projection
        ('x0', 'op_k'),
        ('W_k', 'op_k'),
        ('b_k', 'op_k'),
        ('op_k', 'K'),

        # V projection
        ('x0', 'op_v'),
        ('W_v', 'op_v'),
        ('b_v', 'op_v'),
        ('op_v', 'V'),

        # Attention scores
        ('Q', 'op_qk'),
        ('K', 'op_qk'),
        ('op_qk', 'scores'),
        ('scores', 'op_scale'),
        ('sqrt_d', 'op_scale'),
        ('op_scale', 'op_softmax'),
        ('op_softmax', 'attn'),

        # Attention output
        ('attn', 'op_av'),
        ('V', 'op_av'),
        ('op_av', 'attn_out'),
        ('attn_out', 'op_o'),
        ('W_o', 'op_o'),
        ('b_o', 'op_o'),
        ('op_o', 'proj'),

        # Residual 1
        ('x0', 'op_res1'),
        ('proj', 'op_res1'),
        ('op_res1', 'x1'),

        # FFN layer 1
        ('x1', 'op_ff1'),
        ('W_ff1', 'op_ff1'),
        ('b_ff1', 'op_ff1'),
        ('op_ff1', 'op_relu'),
        ('op_relu', 'h'),

        # FFN layer 2
        ('h', 'op_ff2'),
        ('W_ff2', 'op_ff2'),
        ('b_ff2', 'op_ff2'),
        ('op_ff2', 'ff2'),

        # Residual 2
        ('x1', 'op_res2'),
        ('ff2', 'op_res2'),
        ('op_res2', 'x2'),

        # Output layer
        ('x2', 'op_out'),
        ('W_out', 'op_out'),
        ('b_out', 'op_out'),
        ('op_out', 'logits'),
        ('logits', 'op_logsm'),
        ('op_logsm', 'y_hat'),
    ]

    return nodes, edges, node_info


def save_positions(positions, filepath):
    """Save node positions to a JSON file for manual editing."""
    # Round positions for cleaner JSON
    rounded = {
        node: [round(x, 1), round(y, 1)]
        for node, (x, y) in positions.items()
    }
    with open(filepath, 'w') as f:
        json.dump(rounded, f, indent=2, sort_keys=True)
    print(f"Saved positions to: {filepath}")
    print("  → Edit this file to fine-tune node positions, then re-run the script.")


def load_positions(filepath):
    """Load node positions from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert lists back to tuples
    return {node: tuple(pos) for node, pos in data.items()}


def get_node_style(category):
    """Return Graphviz style attributes for a node category."""
    styles = {
        'param': {
            'shape': 'doublecircle',
            'style': 'filled',
            'fillcolor': '#E8F4FD',
            'width': '0.4',
            'height': '0.4',
            'fixedsize': 'true',
        },
        'op': {
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': '#E8F5E9',
            'width': '0.45',
            'height': '0.45',
            'fixedsize': 'true',
        },
        'activation': {
            'shape': 'doublecircle',
            'style': 'filled',
            'fillcolor': '#FFF3E0',
            'width': '0.4',
            'height': '0.4',
            'fixedsize': 'true',
        },
        'constant': {
            'shape': 'doublecircle',
            'style': 'filled',
            'fillcolor': '#F3E5F5',
            'width': '0.35',
            'height': '0.35',
            'fixedsize': 'true',
        },
    }
    return styles.get(category, styles['op'])


def create_force_directed_graph(width=1000, height=800, iterations=200, seed=42,
                                 positions_file=None):
    """
    Create the Transformer computation graph using Kamada-Kawai layout.

    Args:
        width: Canvas width in points
        height: Canvas height in points
        iterations: Number of layout iterations
        seed: Random seed for reproducibility
        positions_file: Optional JSON file with pre-computed positions

    Returns:
        tuple: (Graphviz Digraph object, positions dict)
    """
    # Get graph definition
    nodes, edges, node_info = define_transformer_graph()

    # Load or compute positions
    if positions_file and os.path.exists(positions_file):
        print(f"Loading positions from: {positions_file}")
        positions = load_positions(positions_file)
    else:
        print(f"Computing Kamada-Kawai layout ({iterations} iterations)...")
        layout = KamadaKawai()
        positions = layout.layout(
            nodes, edges,
            iterations=iterations,
            width=width,
            height=height,
            seed=seed
        )

    # Convert to Graphviz pos attributes (neato uses inches)
    # Scale factor: points to inches (72 points = 1 inch)
    pos_attrs = layout_to_graphviz(positions, scale=1/72)

    # Create Graphviz graph with neato engine (respects pos attribute)
    dot = Digraph(
        comment='Transformer Computation Graph (Force-Directed)',
        engine='neato'
    )

    # Graph attributes
    dot.attr(
        splines='true',  # Curved edges
        overlap='false',
        dpi='150',
    )
    dot.attr('node', fontname='Helvetica', fontsize='9')
    dot.attr('edge', fontname='Helvetica', fontsize='7')

    # Add nodes with computed positions
    for node_id in nodes:
        label, category = node_info[node_id]
        style = get_node_style(category)
        dot.node(
            node_id,
            label,
            pos=pos_attrs[node_id],
            **style
        )

    # Add edges
    for src, tgt in edges:
        dot.edge(src, tgt)

    return dot, positions


def main():
    """Generate and save the force-directed computation graph."""
    parser = argparse.ArgumentParser(
        description='Generate Transformer computation graph with force-directed layout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Run script to generate initial layout:
       python generate_computation_graph_force.py

  2. Open interactive editor to fine-tune positions:
       Open script/interactive_graph_editor.html in a browser
       Drag nodes to adjust positions
       Click "Export JSON" and save to config/graph_positions.json

  3. Re-run script with your custom positions:
       python generate_computation_graph_force.py --positions config/graph_positions.json
        """
    )
    parser.add_argument(
        '--positions', '-p',
        type=str,
        help='JSON file with custom node positions (from interactive editor)'
    )
    parser.add_argument(
        '--save-positions', '-s',
        action='store_true',
        help='Save computed positions to config/graph_positions.json'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=200,
        help='Number of layout iterations (default: 200)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    # Setup directories
    project_dir = os.path.dirname(script_dir)
    media_dir = os.path.join(project_dir, 'media')
    config_dir = os.path.join(project_dir, 'config')
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Positions file path
    positions_file = args.positions
    if positions_file and not os.path.isabs(positions_file):
        positions_file = os.path.join(project_dir, positions_file)

    # Create the computation graph
    print("Generating Transformer computation graph...")
    dot, positions = create_force_directed_graph(
        width=1000,
        height=800,
        iterations=args.iterations,
        seed=args.seed,
        positions_file=positions_file
    )

    # Save positions if requested or if no positions file was provided
    if args.save_positions or (not positions_file):
        positions_out = os.path.join(config_dir, 'graph_positions.json')
        save_positions(positions, positions_out)

    # Add title
    title = 'Transformer Computation Graph'
    if positions_file:
        title += ' (Custom Layout)'
    else:
        title += ' (Kamada-Kawai Layout)'
    dot.attr(
        label=title + '\n',
        labelloc='t',
        fontsize='14',
        fontname='Helvetica-Bold'
    )

    # Output paths
    base_path = os.path.join(media_dir, 'transformer_computation_graph_force')

    # Render to multiple formats
    print(f"Saving to {base_path}.png")
    dot.render(base_path, format='png', cleanup=True)

    print(f"Saving to {base_path}.svg")
    dot.render(base_path, format='svg', cleanup=True)

    print(f"Saving to {base_path}.pdf")
    dot.render(base_path, format='pdf', cleanup=True)

    print("\nDone! Generated files:")
    print(f"  - {base_path}.png")
    print(f"  - {base_path}.svg")
    print(f"  - {base_path}.pdf")

    # Print legend
    print("\nGraph Legend:")
    print("  ⬭ Double circles (blue):   Parameters (W, b)")
    print("  ⬭ Double circles (orange): Activations (x, Q, K, V, h)")
    print("  ○ Single circles (green):  Operations (+, ×, σ, relu, lin)")
    print("  ⬭ Double circles (purple): Constants (√d)")

    # Print next steps
    if not positions_file:
        print("\n" + "="*60)
        print("NEXT STEPS for fine-tuning:")
        print("="*60)
        print("1. Open in browser: script/interactive_graph_editor.html")
        print("2. Drag nodes to adjust positions")
        print("3. Click 'Export JSON' → save to config/graph_positions.json")
        print("4. Re-run: python script/generate_computation_graph_force.py \\")
        print("             --positions config/graph_positions.json")


if __name__ == '__main__':
    main()
