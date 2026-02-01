#!/usr/bin/env python3
"""
Generate an interactive HTML viewer for ONNX computation graphs.

Creates a self-contained HTML file with:
- SVG rendering with Netron-style node styling
- Draggable nodes for manual layout adjustment
- Auto-updating edges that follow nodes
- Export button to save positions as JSON
"""

import os
import json
import onnx


# 4-color palette
COLORS = {
    'io': '#2196F3',           # Blue - Input/Output
    'op': '#37474F',           # Dark Gray - Operations
    'param': '#7B1FA2',        # Purple - Learnable Parameters
    'const': '#FF9800',        # Orange - Constants (non-learnable)
}



def format_shape(shape):
    """Convert shape list to tuple format: [27, 64] -> '(27,64)'."""
    if not shape:
        return ''
    dims = [str(d) if isinstance(d, int) and d > 0 else '?' for d in shape]
    return '(' + ','.join(dims) + ')'


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
    """
    graph = model.graph

    for node in graph.node:
        output_name = node.output[0] if node.output else None
        if not output_name or output_name in shapes:
            continue

        op = node.op_type

        if op == 'Gather':
            if len(node.input) >= 2:
                data_shape = shapes.get(node.input[0], [])
                indices_shape = shapes.get(node.input[1], [])
                if data_shape and indices_shape:
                    shapes[output_name] = indices_shape + data_shape[1:]

        elif op == 'Add':
            for inp in node.input:
                if inp in shapes:
                    shapes[output_name] = shapes[inp]
                    break

        elif op == 'Gemm':
            if len(node.input) >= 2:
                a_shape = shapes.get(node.input[0], [])
                b_shape = shapes.get(node.input[1], [])
                if a_shape and b_shape:
                    trans_b = False
                    for attr in node.attribute:
                        if attr.name == 'transB':
                            trans_b = attr.i == 1
                    if trans_b:
                        out_dim = b_shape[0]
                    else:
                        out_dim = b_shape[1] if len(b_shape) > 1 else b_shape[0]
                    if len(a_shape) > 1:
                        shapes[output_name] = [a_shape[0], out_dim]
                    else:
                        shapes[output_name] = [out_dim]

        elif op == 'Transpose':
            if node.input and node.input[0] in shapes:
                inp_shape = shapes[node.input[0]]
                shapes[output_name] = inp_shape[::-1]

        elif op == 'MatMul':
            if len(node.input) >= 2:
                a_shape = shapes.get(node.input[0], [])
                b_shape = shapes.get(node.input[1], [])
                if a_shape and b_shape:
                    out_shape = [a_shape[0]] if a_shape else []
                    if len(b_shape) > 1:
                        out_shape.append(b_shape[1])
                    elif b_shape:
                        out_shape.append(b_shape[0])
                    shapes[output_name] = out_shape

        elif op == 'Div':
            if node.input and node.input[0] in shapes:
                shapes[output_name] = shapes[node.input[0]]

        elif op in ('Softmax', 'LogSoftmax', 'Relu'):
            if node.input and node.input[0] in shapes:
                shapes[output_name] = shapes[node.input[0]]

        elif op == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    if attr.t.dims:
                        shapes[output_name] = list(attr.t.dims)
                    else:
                        shapes[output_name] = [1]

    return shapes


def get_friendly_tensor_name(tensor_name, context=''):
    """Extract a friendly name from ONNX tensor name."""
    name = tensor_name

    # Check for specific layer components
    if 'embedding' in name.lower():
        if 'weight' in name.lower():
            return 'E'
        return 'e'
    if 'positional' in name.lower():
        if 'weight' in name.lower() or 'encoding' in name.lower():
            return 'PE'
        return 'p'
    if 'W_q' in name or 'w_q' in name.lower():
        return 'Wq'
    if 'W_k' in name or 'w_k' in name.lower():
        return 'Wk'
    if 'W_v' in name or 'w_v' in name.lower():
        return 'Wv'
    if 'W_o' in name or 'w_o' in name.lower():
        return 'Wo'
    if 'ff1' in name.lower():
        if 'weight' in name.lower():
            return 'W1'
        if 'bias' in name.lower():
            return 'b1'
        return 'h1'
    if 'ff2' in name.lower():
        if 'weight' in name.lower():
            return 'W2'
        if 'bias' in name.lower():
            return 'b2'
        return 'h2'
    if 'output_proj' in name.lower():
        if 'weight' in name.lower():
            return 'Wout'
        if 'bias' in name.lower():
            return 'bout'
        return 'out'

    # Generic patterns
    if 'weight' in name.lower():
        return 'W'
    if 'bias' in name.lower():
        return 'b'
    if 'input' in name.lower():
        return 'x'

    # Handle onnx:: prefixed names (like onnx::Add_0, onnx::MatMul_0)
    if name.startswith('onnx::'):
        # onnx::Add_38 with shape (20, 64) is the positional embedding
        if 'Add' in name:
            return 'PE'  # Positional Embedding
        # Extract operation name: onnx::Add_123 -> Add
        op_part = name[6:]  # Remove 'onnx::'
        # Remove trailing numbers and underscores
        import re
        match = re.match(r'([A-Za-z]+)', op_part)
        if match:
            return match.group(1)
        return op_part[:6]

    # Clean up the name
    if name.startswith('model.'):
        name = name[6:]
    if name.endswith('_output_0'):
        name = name[:-9]
    if name.startswith('/model/'):
        name = name[7:]
    name = name.replace('/', '.')
    name = name.replace('layers.0.', '')

    if len(name) > 12:
        name = name[:10] + '..'
    return name


def get_output_var_name(node_name, op_type, output_idx=0):
    """Generate a meaningful output variable name based on the node."""
    name = node_name.lower()

    if 'embedding' in name and 'gather' in name.lower():
        return 'e'
    if 'positional' in name:
        return 'x'
    if 'w_q' in name or '/q/' in name:
        return 'Q'
    if 'w_k' in name or '/k/' in name:
        return 'K'
    if 'w_v' in name or '/v/' in name:
        return 'V'
    if 'w_o' in name:
        return 'a'
    if 'ff1' in name:
        return 'h'
    if 'ff2' in name:
        return 'f'
    if 'output_proj' in name:
        return 'logits'
    if op_type == 'Softmax':
        return 'α'
    if op_type == 'LogSoftmax':
        return 'ŷ'
    if op_type == 'MatMul':
        if 'matmul_1' in name.lower():
            return 'attn'
        return 'scores'
    if op_type == 'Div':
        return 's'
    if op_type == 'Transpose':
        return 'Kᵀ'
    if op_type == 'Add':
        return 'z'
    if op_type == 'Relu':
        return 'h'

    return 'y'


def parse_onnx_model(onnx_path):
    """
    Parse ONNX model and extract nodes, edges, and shapes.
    Returns: (nodes_data, edges_data)
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    # Get tensor shapes
    shapes = get_tensor_shapes(model)
    shapes = infer_output_shapes(model, shapes)

    nodes_data = []
    edges_data = []
    output_to_node = {}  # tensor name -> node id
    initializer_names = {init.name for init in graph.initializer}

    node_idx = 0

    # Create input nodes (non-initializer inputs)
    for inp in graph.input:
        if inp.name not in initializer_names:
            node_id = f'input_{node_idx}'
            shape = shapes.get(inp.name, [])
            var_name = 'x' if 'input' in inp.name.lower() else get_friendly_tensor_name(inp.name)
            nodes_data.append({
                'id': node_id,
                'type': 'Input',
                'op_type': 'Input',
                'formula': f'{var_name} = input',
                'name': inp.name,
                'color': COLORS['io'],
                'text_color': 'white',
                'body_lines': [f'{var_name} {format_shape(shape)}'] if shape else [],
            })
            output_to_node[inp.name] = node_id
            node_idx += 1

    # Create parameter nodes (initializers/weights used by ops)
    created_param_nodes = set()
    for node in graph.node:
        for inp_idx, inp_name in enumerate(node.input):
            if inp_name in initializer_names and inp_name not in created_param_nodes:
                node_id = f'param_{node_idx}'
                shape = shapes.get(inp_name, [])

                # Detect bias vs weight:
                # 1. Check name for 'bias'
                # 2. For Gemm, input[2] is bias
                # 3. 1D tensors are typically biases
                is_bias = False
                if 'bias' in inp_name.lower():
                    is_bias = True
                elif node.op_type == 'Gemm' and inp_idx == 2:
                    is_bias = True
                elif len(shape) == 1:
                    is_bias = True

                var_name = get_friendly_tensor_name(inp_name)
                # Override to 'b' if detected as bias but showing as 'W'
                if is_bias and var_name == 'W':
                    var_name = 'b'

                nodes_data.append({
                    'id': node_id,
                    'type': 'Param',
                    'op_type': 'Param',
                    'formula': var_name,
                    'name': inp_name,
                    'color': COLORS['param'],
                    'text_color': 'white',
                    'body_lines': [format_shape(shape)] if shape else [],
                })
                output_to_node[inp_name] = node_id
                created_param_nodes.add(inp_name)
                node_idx += 1

    # Create operation nodes
    for i, node in enumerate(graph.node):
        node_id = f'op_{node_idx}'
        op_type = node.op_type

        # Get output variable name and shape
        out_tensor = node.output[0] if node.output else ''
        out_shape = shapes.get(out_tensor, [])
        out_var = get_output_var_name(node.name, op_type)

        # Build formula based on operation type
        if op_type == 'Gather':
            data_name = node.input[0] if node.input else ''
            # Token embedding gather: model.embedding.weight
            if 'embedding' in data_name.lower():
                formula = 'e = E[x]'  # Token embedding lookup
            else:
                E_var = get_friendly_tensor_name(data_name)
                formula = f'{out_var} = {E_var}[x]'

        elif op_type == 'Gemm':
            w_name = node.input[1] if len(node.input) > 1 else ''
            W_var = get_friendly_tensor_name(w_name)
            formula = f'{out_var} = X·{W_var} + b'

        elif op_type == 'Add':
            # Check if this is the embedding addition (token + position)
            # In ONNX: Gather_output + onnx::Add_38 (position embedding)
            inp_names = [n.lower() for n in node.input]
            is_embedding_add = (
                any('embedding' in n and 'gather' in n for n in inp_names) or
                any('onnx::add' in n for n in inp_names) or
                'positional_encoding' in (node.output[0].lower() if node.output else '')
            )
            if is_embedding_add:
                formula = 'x = e + p'  # token embedding + position embedding
            else:
                formula = f'{out_var} = A + B'

        elif op_type == 'MatMul':
            a_name = node.input[0] if node.input else ''
            b_name = node.input[1] if len(node.input) > 1 else ''
            # Check what's being multiplied for better naming
            a_lower = a_name.lower()
            b_lower = b_name.lower()

            # Q · K^T for attention scores
            if ('w_q' in a_lower or '/q/' in a_lower) and ('transpose' in b_lower):
                formula = 'scores = Q·Kᵀ'
            # attention_weights · V
            elif 'softmax' in a_lower and ('w_v' in b_lower or '/v/' in b_lower):
                formula = 'attn = α·V'
            else:
                A_var = get_friendly_tensor_name(a_name)
                B_var = get_friendly_tensor_name(b_name)
                formula = f'{out_var} = {A_var}·{B_var}'

        elif op_type == 'Div':
            formula = f'{out_var} = X / √d'

        elif op_type == 'Transpose':
            formula = f'{out_var} = Xᵀ'

        elif op_type == 'Softmax':
            formula = f'{out_var} = softmax(X)'

        elif op_type == 'LogSoftmax':
            formula = f'{out_var} = log_softmax(X)'

        elif op_type == 'Relu':
            formula = f'{out_var} = ReLU(X)'

        elif op_type == 'Constant':
            # Constants get their own color
            # Note: Unicode subscripts (ₖ) don't render well in SVG text
            formula = 's = 1/√d'
            body_lines = ['scalar']
            nodes_data.append({
                'id': node_id,
                'type': 'Const',
                'op_type': op_type,
                'formula': formula,
                'name': node.name,
                'color': COLORS['const'],
                'text_color': 'white',
                'body_lines': body_lines,
            })
            # Map outputs and create edges, then continue
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
            continue  # Skip the normal node creation below

        else:
            formula = f'{out_var} = {op_type}(...)'

        # Body just shows output shape
        body_lines = [f'{out_var} {format_shape(out_shape)}'] if out_shape else []

        nodes_data.append({
            'id': node_id,
            'type': 'Op',
            'op_type': op_type,
            'formula': formula,
            'name': node.name,
            'color': COLORS['op'],
            'text_color': 'white',
            'body_lines': body_lines,
        })

        # Map outputs to this node
        for out in node.output:
            output_to_node[out] = node_id

        # Create edges from inputs (including from parameter nodes)
        for inp_name in node.input:
            if inp_name in output_to_node:
                src_id = output_to_node[inp_name]
                edges_data.append({
                    'source': src_id,
                    'target': node_id,
                    'label': '',  # No labels on edges
                })

        node_idx += 1

    # Create output node
    for out in graph.output:
        node_id = f'output_{node_idx}'
        shape = shapes.get(out.name, [])
        nodes_data.append({
            'id': node_id,
            'type': 'Output',
            'op_type': 'Output',
            'formula': 'ŷ = output',
            'name': out.name,
            'color': COLORS['io'],
            'text_color': 'white',
            'body_lines': [f'ŷ {format_shape(shape)}'] if shape else [],
        })

        # Connect last node to output
        if graph.node:
            last_node = graph.node[-1]
            for out_tensor in last_node.output:
                if out_tensor in output_to_node:
                    edges_data.append({
                        'source': output_to_node[out_tensor],
                        'target': node_id,
                        'label': '',
                    })

        node_idx += 1

    return nodes_data, edges_data


def generate_initial_positions(nodes_data, edges_data):
    """
    Generate node positions using an optimized layered layout.
    - Input nodes are one layer above the operations they feed into
    - Parameters are placed beside the operations that use them
    """
    # Build adjacency structures
    out_neighbors = {n['id']: [] for n in nodes_data}  # children
    in_neighbors = {n['id']: [] for n in nodes_data}   # parents
    node_type_map = {n['id']: n.get('type', 'Op') for n in nodes_data}

    for edge in edges_data:
        if edge['source'] in out_neighbors:
            out_neighbors[edge['source']].append(edge['target'])
        if edge['target'] in in_neighbors:
            in_neighbors[edge['target']].append(edge['source'])

    # Categorize nodes
    input_nodes = [n['id'] for n in nodes_data if n.get('type') == 'Input']
    output_nodes = [n['id'] for n in nodes_data if n.get('type') == 'Output']
    param_nodes = [n['id'] for n in nodes_data if n.get('type') in ('Param', 'Const')]
    op_nodes = [n['id'] for n in nodes_data if n.get('type') == 'Op']

    # Assign layers to operation nodes using longest path (ignoring params/inputs)
    node_layer = {}

    def get_op_layer(node_id, memo={}):
        if node_id in memo:
            return memo[node_id]
        if node_id in param_nodes or node_id in input_nodes:
            return -1  # Will be assigned later
        # Only consider op node parents
        parents = [p for p in in_neighbors[node_id] if p in op_nodes]
        if not parents:
            memo[node_id] = 1  # Start ops at layer 1 (leave room for inputs at layer 0)
        else:
            parent_layers = [get_op_layer(p, memo) for p in parents]
            parent_layers = [l for l in parent_layers if l >= 0]
            memo[node_id] = (max(parent_layers) + 1) if parent_layers else 1
        return memo[node_id]

    for node_id in op_nodes:
        node_layer[node_id] = get_op_layer(node_id)

    # Output nodes go one layer below the last op
    max_op_layer = max(node_layer.values()) if node_layer else 1
    for node_id in output_nodes:
        node_layer[node_id] = max_op_layer + 1

    # Input nodes: one layer above the operations they feed into
    for input_id in input_nodes:
        children = out_neighbors[input_id]
        if children:
            child_layers = [node_layer.get(c, 1) for c in children if c in node_layer]
            if child_layers:
                node_layer[input_id] = min(child_layers) - 1
            else:
                node_layer[input_id] = 0
        else:
            node_layer[input_id] = 0

    # Parameters: one layer above the operation that uses them (same rule as inputs)
    for param_id in param_nodes:
        children = out_neighbors[param_id]
        if children:
            child_layers = [node_layer.get(c, 1) for c in children if c in node_layer]
            if child_layers:
                node_layer[param_id] = min(child_layers) - 1
            else:
                node_layer[param_id] = 0
        else:
            node_layer[param_id] = 0

    # Group nodes by layer
    max_layer = max(node_layer.values()) if node_layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for node_id, layer in node_layer.items():
        if layer >= 0:
            layers[layer].append(node_id)

    # Calculate connectivity for each node (number of edges)
    def get_connectivity(node_id):
        return len(in_neighbors[node_id]) + len(out_neighbors[node_id])

    # Within each layer, sort by connectivity and arrange center-out
    # More connected nodes go to the center
    for layer in layers:
        # Sort by connectivity (highest first)
        sorted_by_conn = sorted(layer, key=get_connectivity, reverse=True)

        # Arrange in center-out pattern: most connected at center,
        # then alternating left and right
        center_out = []
        left = []
        right = []
        for i, node_id in enumerate(sorted_by_conn):
            if i == 0:
                center_out.append(node_id)
            elif i % 2 == 1:
                left.insert(0, node_id)  # Add to left side
            else:
                right.append(node_id)    # Add to right side

        layer.clear()
        layer.extend(left + center_out + right)

    # Position nodes
    positions = {}
    node_width = 170
    h_spacing = 185
    v_spacing = 110

    # Initial placement
    for layer_idx, layer in enumerate(layers):
        for i, node_id in enumerate(layer):
            positions[node_id] = {
                'x': i * h_spacing + 50,
                'y': layer_idx * v_spacing + 50,
            }

    # Iterative refinement for non-parameter nodes
    for iteration in range(5):
        # Forward pass: position ops near their parent ops
        for layer_idx in range(1, len(layers)):
            for node_id in layers[layer_idx]:
                if node_id in param_nodes:
                    continue
                parents = [p for p in in_neighbors[node_id] if p not in param_nodes]
                if parents:
                    avg_x = sum(positions[p]['x'] for p in parents) / len(parents)
                    positions[node_id]['x'] = avg_x

        # Backward pass: adjust parents toward children
        for layer_idx in range(len(layers) - 2, -1, -1):
            for node_id in layers[layer_idx]:
                if node_id in param_nodes:
                    continue
                children = [c for c in out_neighbors[node_id] if c not in param_nodes]
                if children:
                    avg_x = sum(positions[c]['x'] for c in children) / len(children)
                    positions[node_id]['x'] = (positions[node_id]['x'] + avg_x) / 2

        # Position parameters/inputs directly above the operations they feed into
        for param_id in (param_nodes + input_nodes):
            children = out_neighbors[param_id]
            if children:
                child_x = [positions[c]['x'] for c in children if c in positions]
                if child_x:
                    # Center parameter above its children
                    positions[param_id]['x'] = sum(child_x) / len(child_x)

        # Resolve overlaps within each layer
        for layer in layers:
            layer_sorted = sorted(layer, key=lambda n: positions[n]['x'])
            for i in range(1, len(layer_sorted)):
                prev_id = layer_sorted[i - 1]
                curr_id = layer_sorted[i]
                min_x = positions[prev_id]['x'] + h_spacing
                if positions[curr_id]['x'] < min_x:
                    positions[curr_id]['x'] = min_x

    # Center each layer individually around x=500
    canvas_center = 500

    for layer in layers:
        if not layer:
            continue
        layer_x = [positions[n]['x'] for n in layer]
        layer_min = min(layer_x)
        layer_max = max(layer_x)
        layer_center = (layer_min + layer_max + node_width) / 2
        layer_offset = canvas_center - layer_center
        for node_id in layer:
            positions[node_id]['x'] += layer_offset

    return positions


def generate_html(nodes_data, edges_data, positions, output_path):
    """
    Generate the interactive HTML file.
    """
    # Convert data to JSON for embedding
    nodes_json = json.dumps(nodes_data, indent=2)
    edges_json = json.dumps(edges_data, indent=2)
    positions_json = json.dumps(positions, indent=2)

    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Computation Graph Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            overflow: hidden;
        }
        #toolbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 50px;
            background: #2c3e50;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 20px;
            gap: 15px;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        #toolbar h1 {
            font-size: 16px;
            font-weight: 500;
            flex: 1;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #27ae60;
            color: white;
        }
        .btn-primary:hover {
            background: #219a52;
        }
        .btn-secondary {
            background: #3498db;
            color: white;
        }
        .btn-secondary:hover {
            background: #2980b9;
        }
        .btn-warning {
            background: #e67e22;
            color: white;
        }
        .btn-warning:hover {
            background: #d35400;
        }
        #canvas-container {
            position: fixed;
            top: 50px;
            left: 0;
            right: 250px;
            bottom: 0;
            background: white;
            overflow: hidden;
        }
        #graph-canvas {
            width: 100%;
            height: 100%;
        }
        #zoom-controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            display: flex;
            gap: 5px;
            z-index: 10;
        }
        #zoom-controls button {
            width: 36px;
            height: 36px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #zoom-controls button:hover {
            background: #f8f9fa;
        }
        #zoom-level {
            padding: 0 10px;
            display: flex;
            align-items: center;
            font-size: 12px;
            color: #6c757d;
        }
        #sidebar {
            position: fixed;
            top: 50px;
            right: 0;
            width: 250px;
            bottom: 0;
            background: #f8f9fa;
            border-left: 1px solid #dee2e6;
            padding: 15px;
            overflow-y: auto;
        }
        #sidebar h2 {
            font-size: 14px;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        #sidebar h3 {
            font-size: 12px;
            color: #6c757d;
            margin: 15px 0 8px 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 6px 0;
            font-size: 12px;
        }
        .legend-box {
            width: 50px;
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            color: white;
            font-weight: 500;
        }
        .selected-info {
            margin-top: 15px;
            padding: 10px;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-size: 12px;
        }
        .selected-info label {
            display: block;
            color: #6c757d;
            margin-bottom: 3px;
        }
        .selected-info input {
            width: 100%;
            padding: 5px;
            border: 1px solid #ced4da;
            border-radius: 3px;
            font-size: 12px;
            margin-bottom: 8px;
        }
        #instructions {
            margin-top: 15px;
            font-size: 11px;
            color: #6c757d;
            line-height: 1.6;
        }
        #instructions p {
            margin: 4px 0;
        }
        .node-group {
            cursor: grab;
        }
        .node-group:active {
            cursor: grabbing;
        }
        .node-group.selected .node-header {
            stroke: #e74c3c;
            stroke-width: 3px;
        }
        #selection-rect {
            fill: rgba(33, 150, 243, 0.1);
            stroke: #2196F3;
            stroke-width: 1;
            stroke-dasharray: 5,3;
        }
        .edge-path {
            fill: none;
            stroke: #95a5a6;
            stroke-width: 1.5;
        }
        .edge-label {
            font-size: 10px;
            fill: #6c757d;
        }
        #json-output {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 5px 30px rgba(0,0,0,0.3);
            max-width: 600px;
            max-height: 80vh;
            overflow: auto;
            z-index: 200;
            display: none;
        }
        #json-output pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            font-size: 11px;
            max-height: 400px;
            overflow: auto;
        }
        #json-output .close-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #6c757d;
        }
        #overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 150;
            display: none;
        }
    </style>
</head>
<body>
    <div id="toolbar">
        <h1>ONNX Computation Graph Viewer</h1>
        <button class="btn btn-warning" onclick="resetLayout()">Reset Layout</button>
        <button class="btn btn-secondary" onclick="centerGraph()">Center Graph</button>
        <button class="btn btn-primary" onclick="exportPositions()">Export JSON</button>
    </div>

    <div id="canvas-container">
        <svg id="graph-canvas"></svg>
        <div id="zoom-controls">
            <button onclick="zoomIn()" title="Zoom In">+</button>
            <button onclick="zoomOut()" title="Zoom Out">−</button>
            <button onclick="resetView()" title="Reset View">⌂</button>
            <span id="zoom-level">100%</span>
        </div>
    </div>

    <div id="sidebar">
        <h2>Legend</h2>
        <div class="legend-item">
            <div class="legend-box" style="background: #2196F3;">x, ŷ</div>
            <span>Input / Output</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #37474F;">f(x)</div>
            <span>Operations</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #7B1FA2;">W, b</div>
            <span>Learnable Parameters</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #FF9800;">√d</div>
            <span>Constants</span>
        </div>

        <div id="selected-panel" class="selected-info" style="display: none;">
            <h3>Selected Node</h3>
            <label>ID: <span id="sel-id"></span></label>
            <label>Type: <span id="sel-type"></span></label>
            <label>X:</label>
            <input type="number" id="sel-x" onchange="updateSelectedPosition()">
            <label>Y:</label>
            <input type="number" id="sel-y" onchange="updateSelectedPosition()">
        </div>

        <div id="instructions">
            <h3>Instructions</h3>
            <p><strong>Scroll/Swipe</strong> to pan view</p>
            <p><strong>Drag node</strong> to reposition</p>
            <p><strong>Click node</strong> to select</p>
            <p><strong>Shift+Click</strong> to multi-select</p>
            <p><strong>Drag on canvas</strong> to rectangle select</p>
            <p><strong>+/−/⌂</strong> zoom controls</p>
        </div>
    </div>

    <div id="overlay" onclick="closeExport()"></div>
    <div id="json-output">
        <button class="close-btn" onclick="closeExport()">&times;</button>
        <h2>Exported Positions</h2>
        <p style="margin: 10px 0; font-size: 12px; color: #6c757d;">
            Copy this JSON to save your custom layout
        </p>
        <pre id="json-content"></pre>
        <button class="btn btn-primary" style="margin-top: 10px;" onclick="copyToClipboard()">
            Copy to Clipboard
        </button>
    </div>

    <script>
        // Embedded graph data
        const nodesData = ''' + nodes_json + ''';
        const edgesData = ''' + edges_json + ''';
        let positions = ''' + positions_json + ''';
        const originalPositions = JSON.parse(JSON.stringify(positions));

        // State
        let selectedNodes = new Set();  // Multi-select support
        let svg, nodesGroup, edgesGroup, mainGroup;
        let width, height;

        // Pan and Zoom state
        let scale = 1;
        let panX = 0, panY = 0;

        // Drag state for bulk drag
        let isDragging = false;
        let dragStartX = 0, dragStartY = 0;
        let dragStartPositions = {};

        // Rectangle selection state
        let isSelecting = false;
        let selectStartX = 0, selectStartY = 0;
        let selectionRect = null;

        // Node dimensions
        const NODE_WIDTH = 160;
        const NODE_HEADER_HEIGHT = 32;
        const NODE_LINE_HEIGHT = 16;
        const NODE_PADDING = 6;

        function getNodeHeight(node) {
            const bodyLines = node.body_lines.length;
            if (bodyLines === 0) {
                return NODE_HEADER_HEIGHT + NODE_PADDING;
            }
            return NODE_HEADER_HEIGHT + bodyLines * NODE_LINE_HEIGHT + NODE_PADDING * 2;
        }

        function init() {
            svg = document.getElementById('graph-canvas');
            const container = document.getElementById('canvas-container');
            width = container.clientWidth;
            height = container.clientHeight;

            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

            // Create arrow marker
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            defs.innerHTML = `
                <marker id="arrowhead" markerWidth="10" markerHeight="7"
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#95a5a6" />
                </marker>
            `;
            svg.appendChild(defs);

            // Create main group for pan/zoom transforms
            mainGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            mainGroup.id = 'main-group';
            svg.appendChild(mainGroup);

            // Create groups for edges (below) and nodes (above)
            edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            edgesGroup.id = 'edges';
            mainGroup.appendChild(edgesGroup);

            nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            nodesGroup.id = 'nodes';
            mainGroup.appendChild(nodesGroup);

            render();

            // Click on canvas to deselect all (only if not selecting)
            svg.addEventListener('click', (e) => {
                if (isSelecting) return;
                if (e.target === svg || e.target.tagName === 'path') {
                    selectedNodes.clear();
                    updateSelectedPanel();
                    render();
                }
            });

            // Rectangle selection: mousedown on background starts selection
            svg.addEventListener('mousedown', (e) => {
                // Only start selection if clicking on empty area (not on a node)
                if (e.target !== svg && !e.target.classList.contains('edge-path')) return;

                isSelecting = true;
                const rect = svg.getBoundingClientRect();
                // Convert to SVG coordinates accounting for pan and scale
                selectStartX = (e.clientX - rect.left - panX) / scale;
                selectStartY = (e.clientY - rect.top - panY) / scale;

                // Create selection rectangle element
                selectionRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                selectionRect.setAttribute('id', 'selection-rect');
                selectionRect.setAttribute('x', selectStartX);
                selectionRect.setAttribute('y', selectStartY);
                selectionRect.setAttribute('width', 0);
                selectionRect.setAttribute('height', 0);
                mainGroup.appendChild(selectionRect);
            });

            // Rectangle selection: mousemove updates rectangle size
            svg.addEventListener('mousemove', (e) => {
                if (!isSelecting || !selectionRect) return;

                const rect = svg.getBoundingClientRect();
                const currentX = (e.clientX - rect.left - panX) / scale;
                const currentY = (e.clientY - rect.top - panY) / scale;

                // Calculate rectangle bounds (handle negative drag)
                const x = Math.min(selectStartX, currentX);
                const y = Math.min(selectStartY, currentY);
                const width = Math.abs(currentX - selectStartX);
                const height = Math.abs(currentY - selectStartY);

                selectionRect.setAttribute('x', x);
                selectionRect.setAttribute('y', y);
                selectionRect.setAttribute('width', width);
                selectionRect.setAttribute('height', height);
            });

            // Rectangle selection: mouseup finalizes selection
            svg.addEventListener('mouseup', (e) => {
                if (!isSelecting || !selectionRect) return;

                const rect = svg.getBoundingClientRect();
                const currentX = (e.clientX - rect.left - panX) / scale;
                const currentY = (e.clientY - rect.top - panY) / scale;

                // Calculate selection bounds
                const selX = Math.min(selectStartX, currentX);
                const selY = Math.min(selectStartY, currentY);
                const selW = Math.abs(currentX - selectStartX);
                const selH = Math.abs(currentY - selectStartY);

                // Only select if rectangle has meaningful size
                if (selW > 5 && selH > 5) {
                    // If not shift/ctrl, clear previous selection
                    if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
                        selectedNodes.clear();
                    }

                    // Find nodes within selection rectangle
                    nodesData.forEach(node => {
                        const pos = positions[node.id];
                        if (!pos) return;

                        const nodeHeight = getNodeHeight(node);
                        // Check if node overlaps with selection rectangle
                        const nodeLeft = pos.x;
                        const nodeRight = pos.x + NODE_WIDTH;
                        const nodeTop = pos.y;
                        const nodeBottom = pos.y + nodeHeight;

                        const selRight = selX + selW;
                        const selBottom = selY + selH;

                        // Node is within selection if rectangles overlap
                        if (nodeLeft < selRight && nodeRight > selX &&
                            nodeTop < selBottom && nodeBottom > selY) {
                            selectedNodes.add(node.id);
                        }
                    });

                    updateSelectedPanel();
                    render();
                }

                // Remove selection rectangle
                if (selectionRect && selectionRect.parentNode) {
                    selectionRect.parentNode.removeChild(selectionRect);
                }
                selectionRect = null;
                isSelecting = false;
            });

            // Scroll wheel: pan the view (not zoom)
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                // Use deltaX for horizontal scroll, deltaY for vertical scroll
                // Invert for natural scrolling feel
                panX -= e.deltaX;
                panY -= e.deltaY;
                updateTransform();
            });
        }

        function updateTransform() {
            mainGroup.setAttribute('transform', `translate(${panX}, ${panY}) scale(${scale})`);
        }

        function updateZoomLevel() {
            document.getElementById('zoom-level').textContent = Math.round(scale * 100) + '%';
        }

        function zoomIn() {
            scale = Math.min(scale * 1.2, 5);
            updateTransform();
            updateZoomLevel();
        }

        function zoomOut() {
            scale = Math.max(scale * 0.8, 0.1);
            updateTransform();
            updateZoomLevel();
        }

        function resetView() {
            scale = 1;
            panX = 0;
            panY = 0;
            updateTransform();
            updateZoomLevel();
        }

        function render() {
            edgesGroup.innerHTML = '';
            nodesGroup.innerHTML = '';

            // Preserve transform
            updateTransform();

            // Build node lookup for edge calculations
            const nodeMap = {};
            nodesData.forEach(n => {
                nodeMap[n.id] = n;
            });

            // Draw edges first (below nodes)
            edgesData.forEach(edge => {
                const srcNode = nodeMap[edge.source];
                const tgtNode = nodeMap[edge.target];
                const srcPos = positions[edge.source];
                const tgtPos = positions[edge.target];
                if (!srcPos || !tgtPos || !srcNode || !tgtNode) return;

                const srcHeight = getNodeHeight(srcNode);
                const tgtHeight = getNodeHeight(tgtNode);

                // Calculate edge endpoints (center-bottom to center-top)
                const x1 = srcPos.x + NODE_WIDTH / 2;
                const y1 = srcPos.y + srcHeight;
                const x2 = tgtPos.x + NODE_WIDTH / 2;
                const y2 = tgtPos.y - 8; // Space for arrow

                // Create curved path
                const midY = (y1 + y2) / 2;
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const d = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
                path.setAttribute('d', d);
                path.setAttribute('class', 'edge-path');
                path.setAttribute('marker-end', 'url(#arrowhead)');
                edgesGroup.appendChild(path);

                // Edge label
                if (edge.label) {
                    const labelX = (x1 + x2) / 2;
                    const labelY = midY - 5;
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', labelX);
                    text.setAttribute('y', labelY);
                    text.setAttribute('class', 'edge-label');
                    text.setAttribute('text-anchor', 'middle');
                    text.textContent = edge.label;
                    edgesGroup.appendChild(text);
                }
            });

            // Draw nodes
            nodesData.forEach(node => {
                const pos = positions[node.id];
                if (!pos) return;

                const nodeHeight = getNodeHeight(node);
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('class', 'node-group');
                g.setAttribute('data-id', node.id);
                g.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);

                // Node background (body)
                const body = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                body.setAttribute('x', 0);
                body.setAttribute('y', 0);
                body.setAttribute('width', NODE_WIDTH);
                body.setAttribute('height', nodeHeight);
                body.setAttribute('rx', 6);
                body.setAttribute('ry', 6);
                body.setAttribute('fill', 'white');
                body.setAttribute('stroke', '#dee2e6');
                body.setAttribute('stroke-width', 1);
                g.appendChild(body);

                // Header background
                const header = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                header.setAttribute('class', 'node-header');
                header.setAttribute('x', 0);
                header.setAttribute('y', 0);
                header.setAttribute('width', NODE_WIDTH);
                header.setAttribute('height', NODE_HEADER_HEIGHT);
                header.setAttribute('rx', 6);
                header.setAttribute('ry', 6);
                header.setAttribute('fill', node.color);
                g.appendChild(header);

                // Header bottom cover (to square off bottom of header)
                const headerBottom = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                headerBottom.setAttribute('x', 0);
                headerBottom.setAttribute('y', NODE_HEADER_HEIGHT - 6);
                headerBottom.setAttribute('width', NODE_WIDTH);
                headerBottom.setAttribute('height', 6);
                headerBottom.setAttribute('fill', node.color);
                g.appendChild(headerBottom);

                // Header text (formula)
                const headerText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                headerText.setAttribute('x', NODE_WIDTH / 2);
                headerText.setAttribute('y', NODE_HEADER_HEIGHT / 2 + 4);
                headerText.setAttribute('text-anchor', 'middle');
                headerText.setAttribute('fill', node.text_color);
                headerText.setAttribute('font-size', '11');
                headerText.setAttribute('font-weight', '500');
                headerText.setAttribute('font-style', 'italic');
                headerText.textContent = node.formula || node.op_type;
                g.appendChild(headerText);

                // Body text (tensor info)
                node.body_lines.forEach((line, idx) => {
                    const bodyText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    bodyText.setAttribute('x', NODE_PADDING);
                    bodyText.setAttribute('y', NODE_HEADER_HEIGHT + NODE_PADDING + (idx + 1) * NODE_LINE_HEIGHT - 3);
                    bodyText.setAttribute('fill', '#495057');
                    bodyText.setAttribute('font-size', '10');
                    bodyText.setAttribute('font-family', 'monospace');
                    bodyText.textContent = line;
                    g.appendChild(bodyText);
                });

                // Click handling for selection (Shift/Ctrl+Click for multi-select)
                g.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (e.shiftKey || e.ctrlKey || e.metaKey) {
                        // Toggle selection
                        if (selectedNodes.has(node.id)) {
                            selectedNodes.delete(node.id);
                        } else {
                            selectedNodes.add(node.id);
                        }
                    } else {
                        // Single select (unless already selected for drag)
                        if (!selectedNodes.has(node.id)) {
                            selectedNodes.clear();
                            selectedNodes.add(node.id);
                        }
                    }
                    updateSelectedPanel();
                    render();
                });

                // Drag handling for bulk drag
                g.addEventListener('mousedown', (e) => {
                    // If node not selected, select it first
                    if (!selectedNodes.has(node.id)) {
                        if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
                            selectedNodes.clear();
                        }
                        selectedNodes.add(node.id);
                        render();
                    }

                    // Start dragging all selected nodes
                    isDragging = true;
                    dragStartX = e.clientX;
                    dragStartY = e.clientY;
                    // Store starting positions of all selected nodes
                    dragStartPositions = {};
                    selectedNodes.forEach(id => {
                        dragStartPositions[id] = { x: positions[id].x, y: positions[id].y };
                    });
                    e.stopPropagation();
                });

                nodesGroup.appendChild(g);
            });

            // Highlight all selected nodes
            selectedNodes.forEach(nodeId => {
                const nodeEl = nodesGroup.querySelector(`[data-id="${nodeId}"]`);
                if (nodeEl) nodeEl.classList.add('selected');
            });
        }

        // Global mouse move/up for bulk drag
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = (e.clientX - dragStartX) / scale;
            const dy = (e.clientY - dragStartY) / scale;
            // Move all selected nodes
            selectedNodes.forEach(id => {
                if (dragStartPositions[id]) {
                    positions[id].x = dragStartPositions[id].x + dx;
                    positions[id].y = dragStartPositions[id].y + dy;
                }
            });
            render();
            updateSelectedPanel();
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            dragStartPositions = {};
        });

        function updateSelectedPanel() {
            const panel = document.getElementById('selected-panel');

            if (selectedNodes.size === 0) {
                panel.style.display = 'none';
                return;
            }

            panel.style.display = 'block';
            const nodeIds = Array.from(selectedNodes);

            if (selectedNodes.size === 1) {
                const nodeId = nodeIds[0];
                const node = nodesData.find(n => n.id === nodeId);
                const pos = positions[nodeId];
                document.getElementById('sel-id').textContent = nodeId;
                document.getElementById('sel-type').textContent = node ? node.op_type : '';
                document.getElementById('sel-x').value = Math.round(pos.x);
                document.getElementById('sel-y').value = Math.round(pos.y);
            } else {
                document.getElementById('sel-id').textContent = `${selectedNodes.size} nodes`;
                document.getElementById('sel-type').textContent = 'multiple';
                document.getElementById('sel-x').value = '-';
                document.getElementById('sel-y').value = '-';
            }
        }

        function updateSelectedPosition() {
            if (selectedNodes.size !== 1) return;
            const nodeId = Array.from(selectedNodes)[0];
            positions[nodeId].x = parseFloat(document.getElementById('sel-x').value);
            positions[nodeId].y = parseFloat(document.getElementById('sel-y').value);
            render();
        }

        function resetLayout() {
            positions = JSON.parse(JSON.stringify(originalPositions));
            render();
        }

        function centerGraph() {
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;

            Object.entries(positions).forEach(([nodeId, pos]) => {
                const node = nodesData.find(n => n.id === nodeId);
                const h = node ? getNodeHeight(node) : 50;
                minX = Math.min(minX, pos.x);
                maxX = Math.max(maxX, pos.x + NODE_WIDTH);
                minY = Math.min(minY, pos.y);
                maxY = Math.max(maxY, pos.y + h);
            });

            const graphWidth = maxX - minX;
            const graphHeight = maxY - minY;
            const centerX = (width - 250) / 2;
            const centerY = height / 2;
            const offsetX = centerX - (minX + graphWidth / 2);
            const offsetY = centerY - (minY + graphHeight / 2);

            Object.keys(positions).forEach(nodeId => {
                positions[nodeId].x += offsetX;
                positions[nodeId].y += offsetY;
            });

            render();
        }

        function exportPositions() {
            const exportData = {};
            Object.entries(positions).forEach(([nodeId, pos]) => {
                exportData[nodeId] = [Math.round(pos.x), Math.round(pos.y)];
            });

            const json = JSON.stringify(exportData, null, 2);
            document.getElementById('json-content').textContent = json;
            document.getElementById('overlay').style.display = 'block';
            document.getElementById('json-output').style.display = 'block';
        }

        function closeExport() {
            document.getElementById('overlay').style.display = 'none';
            document.getElementById('json-output').style.display = 'none';
        }

        function copyToClipboard() {
            const json = document.getElementById('json-content').textContent;
            navigator.clipboard.writeText(json).then(() => {
                alert('Copied to clipboard!');
            });
        }

        window.addEventListener('load', init);
        window.addEventListener('resize', () => {
            const container = document.getElementById('canvas-container');
            width = container.clientWidth;
            height = container.clientHeight;
            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            render();
        });
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)


def main():
    """Generate the interactive HTML viewer from ONNX model."""

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
    nodes_data, edges_data = parse_onnx_model(onnx_path)

    print(f"Parsed {len(nodes_data)} nodes and {len(edges_data)} edges")

    # Generate initial positions
    positions = generate_initial_positions(nodes_data, edges_data)

    # Generate HTML
    output_path = os.path.join(media_dir, 'part1_interactive_graph.html')
    generate_html(nodes_data, edges_data, positions, output_path)

    print(f"\nGenerated: {output_path}")
    print("\nOpen this file in your browser to view and edit the graph.")
    print("Drag nodes to adjust positions, then click 'Export JSON' to save.")


if __name__ == '__main__':
    main()
