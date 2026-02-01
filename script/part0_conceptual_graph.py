#!/usr/bin/env python3
"""
Generate Part 0: Conceptual Transformer Architecture Interactive Graph.

This script creates an interactive HTML viewer showing the high-level
Transformer architecture following the ONNX interactive graph skill standards.
"""

import os
import json
from collections import defaultdict

# =============================================================================
# 4-COLOR PALETTE (from skill standard)
# =============================================================================

COLORS = {
    'io': '#2196F3',           # Blue - Input/Output
    'op': '#37474F',           # Dark Gray - Operations
    'param': '#7B1FA2',        # Purple - Learnable Parameters
    'const': '#FF9800',        # Orange - Constants
}

# =============================================================================
# CONCEPTUAL TRANSFORMER ARCHITECTURE
# =============================================================================

def create_conceptual_graph():
    """
    Create the conceptual Transformer architecture nodes and edges.
    Returns: (nodes_data, edges_data)
    """
    nodes_data = [
        # Input
        {
            'id': 'input_0',
            'type': 'Input',
            'formula': 'x = input',
            'color': COLORS['io'],
            'text_color': 'white',
            'body_lines': ['x (seq_len,)'],
        },

        # Embedding Matrix (Parameter)
        {
            'id': 'param_E',
            'type': 'Param',
            'formula': 'E',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(vocab,d_model)'],
        },

        # Token Embedding (Operation)
        {
            'id': 'op_embed',
            'type': 'Op',
            'formula': 'e = E[x]',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['e (seq_len,d_model)'],
        },

        # Positional Encoding (Parameter)
        {
            'id': 'param_PE',
            'type': 'Param',
            'formula': 'PE',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(seq_len,d_model)'],
        },

        # Add Positional Encoding
        {
            'id': 'op_pos_add',
            'type': 'Op',
            'formula': 'x = e + PE',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['x (seq_len,d_model)'],
        },

        # === Self-Attention ===

        # Query Weight
        {
            'id': 'param_Wq',
            'type': 'Param',
            'formula': 'Wq',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_model,d_k)'],
        },

        # Key Weight
        {
            'id': 'param_Wk',
            'type': 'Param',
            'formula': 'Wk',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_model,d_k)'],
        },

        # Value Weight
        {
            'id': 'param_Wv',
            'type': 'Param',
            'formula': 'Wv',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_model,d_v)'],
        },

        # Query Projection
        {
            'id': 'op_Q',
            'type': 'Op',
            'formula': 'Q = x·Wq',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['Q (seq_len,d_k)'],
        },

        # Key Projection
        {
            'id': 'op_K',
            'type': 'Op',
            'formula': 'K = x·Wk',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['K (seq_len,d_k)'],
        },

        # Value Projection
        {
            'id': 'op_V',
            'type': 'Op',
            'formula': 'V = x·Wv',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['V (seq_len,d_v)'],
        },

        # Scale Factor (Constant)
        {
            'id': 'const_sqrt',
            'type': 'Const',
            'formula': '√d_k',
            'color': COLORS['const'],
            'text_color': 'white',
            'body_lines': ['scalar'],
        },

        # Attention Scores
        {
            'id': 'op_scores',
            'type': 'Op',
            'formula': 's = Q·Kᵀ / √d_k',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['s (seq_len,seq_len)'],
        },

        # Softmax
        {
            'id': 'op_softmax',
            'type': 'Op',
            'formula': 'α = softmax(s)',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['α (seq_len,seq_len)'],
        },

        # Context
        {
            'id': 'op_context',
            'type': 'Op',
            'formula': 'c = α·V',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['c (seq_len,d_v)'],
        },

        # Output Projection Weight
        {
            'id': 'param_Wo',
            'type': 'Param',
            'formula': 'Wo',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_v,d_model)'],
        },

        # Attention Output
        {
            'id': 'op_attn_out',
            'type': 'Op',
            'formula': 'a = c·Wo',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['a (seq_len,d_model)'],
        },

        # Residual 1
        {
            'id': 'op_res1',
            'type': 'Op',
            'formula': 'z = x + a',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['z (seq_len,d_model)'],
        },

        # === Feed-Forward Network ===

        # FFN Weight 1
        {
            'id': 'param_W1',
            'type': 'Param',
            'formula': 'W1',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_model,d_ff)'],
        },

        # FFN Layer 1
        {
            'id': 'op_ff1',
            'type': 'Op',
            'formula': 'h = ReLU(z·W1)',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['h (seq_len,d_ff)'],
        },

        # FFN Weight 2
        {
            'id': 'param_W2',
            'type': 'Param',
            'formula': 'W2',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_ff,d_model)'],
        },

        # FFN Layer 2
        {
            'id': 'op_ff2',
            'type': 'Op',
            'formula': 'f = h·W2',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['f (seq_len,d_model)'],
        },

        # Residual 2
        {
            'id': 'op_res2',
            'type': 'Op',
            'formula': 'out = z + f',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['out (seq_len,d_model)'],
        },

        # === Output Layer ===

        # Output Weight
        {
            'id': 'param_Wout',
            'type': 'Param',
            'formula': 'Wout',
            'color': COLORS['param'],
            'text_color': 'white',
            'body_lines': ['(d_model,num_classes)'],
        },

        # Logits
        {
            'id': 'op_logits',
            'type': 'Op',
            'formula': 'logits = out·Wout',
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': ['logits (seq_len,num_classes)'],
        },

        # Output
        {
            'id': 'output_0',
            'type': 'Output',
            'formula': 'ŷ = log_softmax(logits)',
            'color': COLORS['io'],
            'text_color': 'white',
            'body_lines': ['ŷ (seq_len,num_classes)'],
        },
    ]

    edges_data = [
        # Input to Embedding
        {'source': 'input_0', 'target': 'op_embed'},
        {'source': 'param_E', 'target': 'op_embed'},

        # Embedding to Positional Add
        {'source': 'op_embed', 'target': 'op_pos_add'},
        {'source': 'param_PE', 'target': 'op_pos_add'},

        # Positional to Q, K, V
        {'source': 'op_pos_add', 'target': 'op_Q'},
        {'source': 'op_pos_add', 'target': 'op_K'},
        {'source': 'op_pos_add', 'target': 'op_V'},
        {'source': 'param_Wq', 'target': 'op_Q'},
        {'source': 'param_Wk', 'target': 'op_K'},
        {'source': 'param_Wv', 'target': 'op_V'},

        # Q, K to Scores
        {'source': 'op_Q', 'target': 'op_scores'},
        {'source': 'op_K', 'target': 'op_scores'},
        {'source': 'const_sqrt', 'target': 'op_scores'},

        # Scores to Softmax
        {'source': 'op_scores', 'target': 'op_softmax'},

        # Softmax + V to Context
        {'source': 'op_softmax', 'target': 'op_context'},
        {'source': 'op_V', 'target': 'op_context'},

        # Context to Attention Output
        {'source': 'op_context', 'target': 'op_attn_out'},
        {'source': 'param_Wo', 'target': 'op_attn_out'},

        # Attention Output + Residual
        {'source': 'op_pos_add', 'target': 'op_res1'},
        {'source': 'op_attn_out', 'target': 'op_res1'},

        # FFN
        {'source': 'op_res1', 'target': 'op_ff1'},
        {'source': 'param_W1', 'target': 'op_ff1'},
        {'source': 'op_ff1', 'target': 'op_ff2'},
        {'source': 'param_W2', 'target': 'op_ff2'},

        # FFN + Residual
        {'source': 'op_res1', 'target': 'op_res2'},
        {'source': 'op_ff2', 'target': 'op_res2'},

        # Output
        {'source': 'op_res2', 'target': 'op_logits'},
        {'source': 'param_Wout', 'target': 'op_logits'},
        {'source': 'op_logits', 'target': 'output_0'},
    ]

    return nodes_data, edges_data


def compute_layout(nodes_data, edges_data):
    """
    Compute topological layout with layers.
    - Input/Parameter/Constant nodes: one layer above operations they feed into
    - Operations flow top-to-bottom
    - Center-align all layers
    """
    # Build adjacency
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    for edge in edges_data:
        outgoing[edge['source']].add(edge['target'])
        incoming[edge['target']].add(edge['source'])

    node_ids = [n['id'] for n in nodes_data]
    node_types = {n['id']: n['type'] for n in nodes_data}

    # Separate by type
    input_nodes = [n for n in node_ids if node_types[n] == 'Input']
    output_nodes = [n for n in node_ids if node_types[n] == 'Output']
    param_nodes = [n for n in node_ids if node_types[n] in ('Param', 'Const')]
    op_nodes = [n for n in node_ids if node_types[n] == 'Op']

    # Assign layers to operations using topological order
    layers = {}

    def get_op_layer(node_id, memo={}):
        if node_id in memo:
            return memo[node_id]
        if node_id in param_nodes or node_id in input_nodes:
            return -1
        parents = [p for p in incoming[node_id] if p in op_nodes]
        if not parents:
            memo[node_id] = 1
        else:
            parent_layers = [get_op_layer(p, memo) for p in parents]
            parent_layers = [l for l in parent_layers if l >= 0]
            memo[node_id] = (max(parent_layers) + 1) if parent_layers else 1
        return memo[node_id]

    for node_id in op_nodes:
        layers[node_id] = get_op_layer(node_id)

    # Output nodes one layer below last op
    max_op_layer = max(layers.values()) if layers else 1
    for node_id in output_nodes:
        layers[node_id] = max_op_layer + 1

    # Input/Param nodes one layer above their children
    for node_id in input_nodes + param_nodes:
        children = outgoing[node_id]
        if children:
            child_layers = [layers.get(c, 1) for c in children if c in layers]
            if child_layers:
                layers[node_id] = min(child_layers) - 1
            else:
                layers[node_id] = 0
        else:
            layers[node_id] = 0

    # Group nodes by layer
    max_layer = max(layers.values()) if layers else 0
    layer_nodes = defaultdict(list)
    for node_id, layer in layers.items():
        layer_nodes[layer].append(node_id)

    # Sort by connectivity (more connected = center)
    def get_connectivity(node_id):
        return len(incoming[node_id]) + len(outgoing[node_id])

    for l in layer_nodes:
        sorted_nodes = sorted(layer_nodes[l], key=get_connectivity, reverse=True)
        # Center-out arrangement
        center_out = []
        left = []
        right = []
        for i, node_id in enumerate(sorted_nodes):
            if i == 0:
                center_out.append(node_id)
            elif i % 2 == 1:
                left.insert(0, node_id)
            else:
                right.append(node_id)
        layer_nodes[l] = left + center_out + right

    # Assign positions
    positions = {}
    node_width = 160
    h_spacing = 200
    v_spacing = 100

    for l in range(min(layer_nodes.keys()), max(layer_nodes.keys()) + 1):
        nodes_in_layer = layer_nodes[l]
        n_nodes = len(nodes_in_layer)
        start_x = 600 - (n_nodes - 1) * h_spacing / 2
        for i, node_id in enumerate(nodes_in_layer):
            positions[node_id] = {
                'x': start_x + i * h_spacing,
                'y': 80 + (l - min(layer_nodes.keys())) * v_spacing,
            }

    return positions


def generate_html(nodes_data, edges_data, positions, output_path):
    """Generate the interactive HTML file following skill standards."""

    # Prepare node data with positions
    for node in nodes_data:
        pos = positions.get(node['id'], {'x': 100, 'y': 100})
        node['x'] = pos['x']
        node['y'] = pos['y']

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Part 0: Transformer Architecture - Interactive Graph</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }
        #container {
            position: fixed;
            top: 0;
            left: 0;
            width: calc(100% - 280px);
            height: 100%;
            overflow: hidden;
            cursor: grab;
        }
        #container:active { cursor: grabbing; }
        #graph {
            transform-origin: 0 0;
        }
        .node {
            position: absolute;
            border-radius: 8px;
            cursor: move;
            user-select: none;
            min-width: 140px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: box-shadow 0.15s;
        }
        .node:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
        .node.selected { box-shadow: 0 0 0 3px #ff5252, 0 6px 20px rgba(0,0,0,0.4); }
        .node-header {
            padding: 8px 12px;
            font-weight: 600;
            font-size: 14px;
            border-radius: 8px 8px 0 0;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .node-body {
            padding: 8px 12px;
            font-size: 12px;
            text-align: center;
            border-radius: 0 0 8px 8px;
            background: rgba(0,0,0,0.15);
        }
        #sidebar {
            position: fixed;
            right: 0;
            top: 0;
            width: 280px;
            height: 100%;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #333;
        }
        h2 { font-size: 18px; margin-bottom: 16px; color: #fff; }
        h3 { font-size: 14px; margin: 16px 0 8px; color: #aaa; }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 13px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }
        .info-panel {
            background: #1a1a2e;
            padding: 12px;
            border-radius: 8px;
            margin-top: 12px;
            font-size: 13px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
        }
        .info-label { color: #888; }
        .info-value { color: #fff; font-family: monospace; }
        .controls {
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }
        button {
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            flex: 1;
        }
        button:hover { background: #1976D2; }
        .zoom-controls {
            display: flex;
            gap: 4px;
            margin-top: 12px;
        }
        .zoom-btn {
            width: 36px;
            height: 36px;
            font-size: 20px;
            padding: 0;
            flex: none;
        }
        #selection-rect {
            position: absolute;
            border: 2px dashed #2196F3;
            background: rgba(33, 150, 243, 0.1);
            pointer-events: none;
            display: none;
        }
        svg {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            overflow: visible;
        }
        .edge {
            fill: none;
            stroke: #666;
            stroke-width: 2;
        }
        .edge-arrow {
            fill: #666;
        }
        .instructions {
            font-size: 12px;
            color: #888;
            margin-top: 16px;
            line-height: 1.6;
        }
        .instructions li { margin: 4px 0; }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph">
            <svg id="edges-svg"></svg>
            <div id="nodes-container"></div>
            <div id="selection-rect"></div>
        </div>
    </div>
    <div id="sidebar">
        <h2>Part 0: Transformer Architecture</h2>

        <h3>Legend</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #2196F3;"></div>
            <span>Input / Output</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #37474F;"></div>
            <span>Operations</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #7B1FA2;"></div>
            <span>Learnable Parameters</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF9800;"></div>
            <span>Constants</span>
        </div>

        <h3>Selection</h3>
        <div class="info-panel" id="selection-info">
            <div class="info-row">
                <span class="info-label">Node:</span>
                <span class="info-value" id="info-id">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">Type:</span>
                <span class="info-value" id="info-type">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">X:</span>
                <span class="info-value" id="info-x">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">Y:</span>
                <span class="info-value" id="info-y">-</span>
            </div>
        </div>

        <div class="zoom-controls">
            <button class="zoom-btn" onclick="zoomIn()">+</button>
            <button class="zoom-btn" onclick="zoomOut()">−</button>
            <button class="zoom-btn" onclick="resetView()">⌂</button>
        </div>

        <div class="controls">
            <button onclick="exportPositions()">Export JSON</button>
        </div>

        <h3>Instructions</h3>
        <ul class="instructions">
            <li>Scroll to pan the view</li>
            <li>Drag nodes to reposition</li>
            <li>Shift+Click to multi-select</li>
            <li>Drag on canvas to box-select</li>
            <li>Use +/− buttons to zoom</li>
        </ul>
    </div>

    <script>
        // Data
        const nodesData = ''' + json.dumps(nodes_data, indent=2) + ''';
        const edgesData = ''' + json.dumps(edges_data, indent=2) + ''';

        // State
        let scale = 1;
        let panX = 0;
        let panY = 0;
        let selectedNodes = new Set();
        let isDragging = false;
        let isSelecting = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let selectStartX = 0;
        let selectStartY = 0;
        let draggedNode = null;
        let nodeStartPositions = {};

        const container = document.getElementById('container');
        const graph = document.getElementById('graph');
        const nodesContainer = document.getElementById('nodes-container');
        const edgesSvg = document.getElementById('edges-svg');
        const selectionRect = document.getElementById('selection-rect');

        // Build node lookup
        const nodeById = {};
        nodesData.forEach(n => nodeById[n.id] = n);

        function updateTransform() {
            graph.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        }

        function zoomIn() {
            scale = Math.min(scale * 1.2, 5);
            updateTransform();
        }

        function zoomOut() {
            scale = Math.max(scale * 0.8, 0.1);
            updateTransform();
        }

        function resetView() {
            scale = 1;
            panX = 0;
            panY = 0;
            updateTransform();
        }

        // Scroll to pan (NOT zoom)
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            panX -= e.deltaX;
            panY -= e.deltaY;
            updateTransform();
        }, { passive: false });

        // Create nodes
        nodesData.forEach(node => {
            const el = document.createElement('div');
            el.className = 'node';
            el.id = 'node-' + node.id;
            el.style.left = node.x + 'px';
            el.style.top = node.y + 'px';
            el.style.background = node.color;
            el.innerHTML = `
                <div class="node-header" style="color: ${node.text_color}">${node.formula}</div>
                <div class="node-body" style="color: ${node.text_color}">${node.body_lines.join('<br>')}</div>
            `;

            el.addEventListener('mousedown', (e) => {
                e.stopPropagation();

                if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    // Toggle selection
                    if (selectedNodes.has(node.id)) {
                        selectedNodes.delete(node.id);
                        el.classList.remove('selected');
                    } else {
                        selectedNodes.add(node.id);
                        el.classList.add('selected');
                    }
                    updateSelectionInfo();
                } else {
                    // If clicking unselected node, clear selection and select this one
                    if (!selectedNodes.has(node.id)) {
                        clearSelection();
                        selectedNodes.add(node.id);
                        el.classList.add('selected');
                    }

                    // Start dragging all selected nodes
                    isDragging = true;
                    draggedNode = node.id;
                    dragStartX = e.clientX;
                    dragStartY = e.clientY;

                    // Store starting positions
                    nodeStartPositions = {};
                    selectedNodes.forEach(id => {
                        const n = nodeById[id];
                        nodeStartPositions[id] = { x: n.x, y: n.y };
                    });
                }
                updateSelectionInfo();
            });

            nodesContainer.appendChild(el);
        });

        // Mouse move for dragging
        document.addEventListener('mousemove', (e) => {
            if (isDragging && draggedNode) {
                const dx = (e.clientX - dragStartX) / scale;
                const dy = (e.clientY - dragStartY) / scale;

                selectedNodes.forEach(id => {
                    const node = nodeById[id];
                    const startPos = nodeStartPositions[id];
                    node.x = startPos.x + dx;
                    node.y = startPos.y + dy;

                    const el = document.getElementById('node-' + id);
                    el.style.left = node.x + 'px';
                    el.style.top = node.y + 'px';
                });

                updateEdges();
                updateSelectionInfo();
            }

            if (isSelecting) {
                const rect = container.getBoundingClientRect();
                const currentX = (e.clientX - rect.left - panX) / scale;
                const currentY = (e.clientY - rect.top - panY) / scale;

                const x = Math.min(selectStartX, currentX);
                const y = Math.min(selectStartY, currentY);
                const w = Math.abs(currentX - selectStartX);
                const h = Math.abs(currentY - selectStartY);

                selectionRect.style.display = 'block';
                selectionRect.style.left = x + 'px';
                selectionRect.style.top = y + 'px';
                selectionRect.style.width = w + 'px';
                selectionRect.style.height = h + 'px';
            }
        });

        // Mouse up
        document.addEventListener('mouseup', (e) => {
            if (isSelecting) {
                const rect = container.getBoundingClientRect();
                const currentX = (e.clientX - rect.left - panX) / scale;
                const currentY = (e.clientY - rect.top - panY) / scale;

                const x1 = Math.min(selectStartX, currentX);
                const y1 = Math.min(selectStartY, currentY);
                const x2 = Math.max(selectStartX, currentX);
                const y2 = Math.max(selectStartY, currentY);

                // Select nodes in rectangle
                if (!e.shiftKey) clearSelection();

                nodesData.forEach(node => {
                    const nodeEl = document.getElementById('node-' + node.id);
                    const nodeW = nodeEl.offsetWidth;
                    const nodeH = nodeEl.offsetHeight;

                    if (node.x + nodeW > x1 && node.x < x2 &&
                        node.y + nodeH > y1 && node.y < y2) {
                        selectedNodes.add(node.id);
                        nodeEl.classList.add('selected');
                    }
                });

                selectionRect.style.display = 'none';
                isSelecting = false;
                updateSelectionInfo();
            }

            isDragging = false;
            draggedNode = null;
        });

        // Click on container to start selection or clear
        container.addEventListener('mousedown', (e) => {
            if (e.target === container || e.target === graph || e.target === nodesContainer) {
                const rect = container.getBoundingClientRect();
                selectStartX = (e.clientX - rect.left - panX) / scale;
                selectStartY = (e.clientY - rect.top - panY) / scale;
                isSelecting = true;
            }
        });

        function clearSelection() {
            selectedNodes.forEach(id => {
                const el = document.getElementById('node-' + id);
                if (el) el.classList.remove('selected');
            });
            selectedNodes.clear();
        }

        function updateSelectionInfo() {
            const infoId = document.getElementById('info-id');
            const infoType = document.getElementById('info-type');
            const infoX = document.getElementById('info-x');
            const infoY = document.getElementById('info-y');

            if (selectedNodes.size === 0) {
                infoId.textContent = '-';
                infoType.textContent = '-';
                infoX.textContent = '-';
                infoY.textContent = '-';
            } else if (selectedNodes.size === 1) {
                const id = Array.from(selectedNodes)[0];
                const node = nodeById[id];
                infoId.textContent = id;
                infoType.textContent = node.type;
                infoX.textContent = Math.round(node.x);
                infoY.textContent = Math.round(node.y);
            } else {
                infoId.textContent = selectedNodes.size + ' nodes';
                infoType.textContent = 'multiple';
                infoX.textContent = '-';
                infoY.textContent = '-';
            }
        }

        function updateEdges() {
            edgesSvg.innerHTML = '';

            // Arrow marker
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            defs.innerHTML = `
                <marker id="arrowhead" markerWidth="10" markerHeight="7"
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" class="edge-arrow"/>
                </marker>
            `;
            edgesSvg.appendChild(defs);

            edgesData.forEach(edge => {
                const sourceNode = nodeById[edge.source];
                const targetNode = nodeById[edge.target];
                if (!sourceNode || !targetNode) return;

                const sourceEl = document.getElementById('node-' + edge.source);
                const targetEl = document.getElementById('node-' + edge.target);

                const x1 = sourceNode.x + sourceEl.offsetWidth / 2;
                const y1 = sourceNode.y + sourceEl.offsetHeight;
                const x2 = targetNode.x + targetEl.offsetWidth / 2;
                const y2 = targetNode.y;

                // Bezier curve
                const midY = (y1 + y2) / 2;
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
                path.setAttribute('class', 'edge');
                path.setAttribute('marker-end', 'url(#arrowhead)');
                edgesSvg.appendChild(path);
            });
        }

        function exportPositions() {
            const positions = {};
            nodesData.forEach(n => {
                positions[n.id] = { x: Math.round(n.x), y: Math.round(n.y) };
            });
            const blob = new Blob([JSON.stringify(positions, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'part0_graph_positions.json';
            a.click();
            URL.revokeObjectURL(url);
        }

        // Initial render
        updateEdges();
        updateTransform();
    </script>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated: {output_path}")


def main():
    """Generate the Part 0 conceptual Transformer architecture graph."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'media', 'part0_interactive_graph.html')

    print("Creating Part 0: Conceptual Transformer Architecture...")
    nodes_data, edges_data = create_conceptual_graph()
    print(f"Created {len(nodes_data)} nodes and {len(edges_data)} edges")

    positions = compute_layout(nodes_data, edges_data)
    generate_html(nodes_data, edges_data, positions, output_path)

    print(f"\nTo view the graph:")
    print(f"  cd {os.path.dirname(output_path)} && python -m http.server 8080")
    print(f"  Open: http://localhost:8080/part0_interactive_graph.html")


if __name__ == '__main__':
    main()
