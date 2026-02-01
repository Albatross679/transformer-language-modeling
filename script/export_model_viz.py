"""
Export Part 1 Transformer model to ONNX and generate Netron visualization.
"""
import torch
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import Transformer

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('media', exist_ok=True)

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
with open(config_path) as f:
    config = json.load(f)['part1']['model']

print("Model configuration:")
print(f"  vocab_size: {config['vocab_size']}")
print(f"  num_positions: {config['num_positions']}")
print(f"  d_model: {config['d_model']}")
print(f"  d_internal: {config['d_internal']}")
print(f"  num_classes: {config['num_classes']}")
print(f"  num_layers: {config['num_layers']}")

# Instantiate model
model = Transformer(
    vocab_size=config['vocab_size'],
    num_positions=config['num_positions'],
    d_model=config['d_model'],
    d_internal=config['d_internal'],
    num_classes=config['num_classes'],
    num_layers=config['num_layers']
)
model.eval()

print(f"\nModel structure:")
print(model)

# Create dummy input (seq_len=20)
dummy_input = torch.randint(0, config['vocab_size'], (config['num_positions'],))

# Test forward pass
with torch.no_grad():
    log_probs, attn_maps = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Number of attention maps: {len(attn_maps)}")
    print(f"  Attention map shape: {attn_maps[0].shape}")

# Create a wrapper model that only returns log_probs for ONNX export
class TransformerONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, indices):
        log_probs, _ = self.model(indices)
        return log_probs

wrapper_model = TransformerONNX(model)
wrapper_model.eval()

# Export to ONNX
onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "transformer_part1.onnx")
print(f"\nExporting to ONNX...")

torch.onnx.export(
    wrapper_model,
    dummy_input,
    onnx_path,
    input_names=['input_indices'],
    output_names=['log_probs'],
    opset_version=14,
    verbose=False
)
print(f"Exported ONNX model to: {onnx_path}")

# Print ONNX model info
import onnx

onnx_model = onnx.load(onnx_path)
print(f"\nONNX Model Info:")
print(f"  IR version: {onnx_model.ir_version}")
print(f"  Producer: {onnx_model.producer_name}")
print(f"  Graph name: {onnx_model.graph.name}")
print(f"  Number of nodes: {len(onnx_model.graph.node)}")

# Print model graph nodes summary
print(f"\nGraph Nodes (operations):")
for i, node in enumerate(onnx_model.graph.node):
    print(f"  [{i}] {node.op_type}: {node.name}")

# Generate text-based architecture summary
summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "transformer_part1_architecture.txt")
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("Part 1 Transformer Architecture Summary\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"  vocab_size: {config['vocab_size']}\n")
    f.write(f"  num_positions: {config['num_positions']}\n")
    f.write(f"  d_model: {config['d_model']}\n")
    f.write(f"  d_internal: {config['d_internal']}\n")
    f.write(f"  num_classes: {config['num_classes']}\n")
    f.write(f"  num_layers: {config['num_layers']}\n\n")

    f.write("Model Structure:\n")
    f.write(str(model) + "\n\n")

    f.write("ONNX Graph Nodes:\n")
    for i, node in enumerate(onnx_model.graph.node):
        f.write(f"  [{i:2d}] {node.op_type:15s}: {node.name}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("To visualize interactively, run:\n")
    f.write(f"  netron {onnx_path}\n")
    f.write("=" * 70 + "\n")

print(f"\nGenerated architecture summary: {summary_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  1. ONNX model: {onnx_path}")
print(f"  2. Architecture summary: {summary_path}")
print(f"\nTo visualize the model interactively with Netron, run:")
print(f"  netron model/transformer_part1.onnx")
print("\nThis will open a web browser with an interactive visualization.")
