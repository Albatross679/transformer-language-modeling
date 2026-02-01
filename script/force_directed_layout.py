#!/usr/bin/env python3
"""
Force-Directed Graph Layout Algorithms

Implements spring embedder algorithms for positioning graph nodes:
1. Basic Force-Directed (Spring Embedder)
2. Fruchterman-Reingold (1991)
3. Kamada-Kawai (1989)

Mathematical Foundation:
- Attractive forces (Hooke's Law): F_a = k × (d - d_0)
- Repulsive forces (Coulomb's Law): F_r = k² / d²
- Energy minimization: E = Σ attractive + Σ repulsive

References:
- Fruchterman & Reingold (1991): Graph drawing by force-directed placement
- Kamada & Kawai (1989): An algorithm for drawing general undirected graphs
- Kobourov (2012): Spring Embedders and Force Directed Graph Drawing Algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import math


class ForceDirectedLayout:
    """
    Force-directed graph layout using spring embedder algorithm.

    Mathematical Model:
    - Attractive: F_a = k_a × (d - d_0) for edges (Hooke's Law)
    - Repulsive:  F_r = k_r / d² for all node pairs (Coulomb's Law)
    - Energy:     E = Σ (k_a/2)(d-d_0)² + Σ k_r/d

    Attributes:
        k_a: Spring constant (attractive force strength)
        k_r: Repulsion constant
        damping: Velocity damping factor (0-1)
        ideal_length: Ideal/rest length for springs
        min_dist: Minimum distance to avoid numerical instability
    """

    def __init__(
        self,
        k_attractive: float = 0.1,
        k_repulsive: float = 1000.0,
        damping: float = 0.9,
        ideal_length: float = 100.0,
        min_distance: float = 1.0,
    ):
        self.k_a = k_attractive
        self.k_r = k_repulsive
        self.damping = damping
        self.ideal_length = ideal_length
        self.min_dist = min_distance

    def layout(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        iterations: int = 500,
        initial_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        width: float = 800.0,
        height: float = 600.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute force-directed layout for a graph.

        Args:
            nodes: List of node identifiers
            edges: List of (source, target) tuples
            iterations: Number of simulation iterations
            initial_positions: Optional starting positions
            width, height: Canvas dimensions for initialization
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(nodes)}

        # Initialize positions
        if initial_positions:
            pos = np.array([
                initial_positions.get(node, (np.random.uniform(0, width),
                                              np.random.uniform(0, height)))
                for node in nodes
            ], dtype=float)
        else:
            # Initialize in a circle for better convergence
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = min(width, height) / 3
            pos = np.column_stack([
                width/2 + radius * np.cos(angles),
                height/2 + radius * np.sin(angles)
            ])

        # Initialize velocities
        vel = np.zeros((n, 2))

        # Store original damping
        current_damping = self.damping

        # Simulation loop
        dt = 0.1
        for iteration in range(iterations):
            forces = np.zeros((n, 2))

            # Repulsive forces (all pairs) - O(n²)
            for i in range(n):
                for j in range(i + 1, n):
                    delta = pos[j] - pos[i]
                    dist = max(np.linalg.norm(delta), self.min_dist)

                    # Coulomb repulsion: F = k_r / d²
                    force_magnitude = self.k_r / (dist * dist)
                    force_direction = delta / dist

                    forces[i] -= force_magnitude * force_direction
                    forces[j] += force_magnitude * force_direction

            # Attractive forces (edges only) - O(|E|)
            for src, tgt in edges:
                if src not in node_idx or tgt not in node_idx:
                    continue
                i, j = node_idx[src], node_idx[tgt]

                delta = pos[j] - pos[i]
                dist = max(np.linalg.norm(delta), self.min_dist)

                # Hooke's law: F = k_a × (d - d_0)
                displacement = dist - self.ideal_length
                force_magnitude = self.k_a * displacement
                force_direction = delta / dist

                forces[i] += force_magnitude * force_direction
                forces[j] -= force_magnitude * force_direction

            # Update velocities and positions
            vel = current_damping * (vel + dt * forces)
            pos = pos + dt * vel

            # Cooling: reduce damping over time
            current_damping *= 0.999

        return {node: (pos[i, 0], pos[i, 1]) for i, node in enumerate(nodes)}

    def compute_energy(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        positions: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Compute total energy of the current layout.

        E = Σ (k_a/2)(d - d_0)² + Σ k_r/d

        Lower energy = better layout.
        """
        node_idx = {node: i for i, node in enumerate(nodes)}
        pos = np.array([positions[node] for node in nodes])
        n = len(nodes)

        energy = 0.0

        # Repulsive energy
        for i in range(n):
            for j in range(i + 1, n):
                dist = max(np.linalg.norm(pos[j] - pos[i]), self.min_dist)
                energy += self.k_r / dist

        # Attractive energy
        for src, tgt in edges:
            if src in node_idx and tgt in node_idx:
                i, j = node_idx[src], node_idx[tgt]
                dist = np.linalg.norm(pos[j] - pos[i])
                energy += 0.5 * self.k_a * (dist - self.ideal_length) ** 2

        return energy


class FruchtermanReingold:
    """
    Fruchterman-Reingold algorithm (1991) with temperature-based cooling.

    Key idea: Use simulated annealing with temperature that decreases
    over iterations, allowing large movements early and fine-tuning later.

    Forces:
    - Attractive: f_a(d) = d² / k  (only for edges)
    - Repulsive:  f_r(d) = k² / d  (for all pairs)

    Where k = sqrt(area / num_nodes) is the optimal vertex distance.
    """

    def __init__(self, cooling_factor: float = 0.95):
        self.cooling = cooling_factor

    def layout(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        iterations: int = 100,
        width: float = 800.0,
        height: float = 600.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute Fruchterman-Reingold layout."""
        if seed is not None:
            np.random.seed(seed)

        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(nodes)}
        area = width * height
        k = math.sqrt(area / n)  # Optimal distance

        # Initialize positions randomly
        pos = np.random.rand(n, 2) * [width, height]

        # Initial temperature (maximum displacement per iteration)
        t = width / 10

        for iteration in range(iterations):
            disp = np.zeros((n, 2))  # Displacement vectors

            # Repulsive forces (all pairs)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    delta = pos[i] - pos[j]
                    dist = max(np.linalg.norm(delta), 0.01)
                    # f_r = k² / d
                    disp[i] += (delta / dist) * (k * k / dist)

            # Attractive forces (edges only)
            for src, tgt in edges:
                if src not in node_idx or tgt not in node_idx:
                    continue
                i, j = node_idx[src], node_idx[tgt]
                delta = pos[i] - pos[j]
                dist = max(np.linalg.norm(delta), 0.01)
                # f_a = d² / k
                force = (delta / dist) * (dist * dist / k)
                disp[i] -= force
                disp[j] += force

            # Apply displacement limited by temperature
            for i in range(n):
                disp_norm = np.linalg.norm(disp[i])
                if disp_norm > 0:
                    # Limit displacement to temperature
                    pos[i] += (disp[i] / disp_norm) * min(disp_norm, t)

                # Keep within bounds
                pos[i, 0] = max(0, min(width, pos[i, 0]))
                pos[i, 1] = max(0, min(height, pos[i, 1]))

            # Cool down temperature
            t *= self.cooling

        return {node: (pos[i, 0], pos[i, 1]) for i, node in enumerate(nodes)}


class KamadaKawai:
    """
    Kamada-Kawai algorithm (1989) based on stress minimization.

    Key idea: Position nodes so that Euclidean distances approximate
    graph-theoretic distances (shortest paths).

    Minimizes: Stress = Σ w_ij × (||p_i - p_j|| - d_ij)²

    Where:
    - d_ij = shortest path length between nodes i and j
    - w_ij = 1/d_ij² (weight inversely proportional to distance)

    This produces layouts that preserve graph structure well.
    """

    def __init__(self):
        pass

    def layout(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        iterations: int = 100,
        width: float = 800.0,
        height: float = 600.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute Kamada-Kawai layout using stress minimization."""
        if seed is not None:
            np.random.seed(seed)

        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(nodes)}

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for src, tgt in edges:
            if src in node_idx and tgt in node_idx:
                i, j = node_idx[src], node_idx[tgt]
                adj[i].append(j)
                adj[j].append(i)

        # Compute all-pairs shortest paths (BFS)
        d = np.full((n, n), float('inf'))
        for i in range(n):
            d[i, i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if d[i, v] == float('inf'):
                        d[i, v] = d[i, u] + 1
                        queue.append(v)

        # Handle disconnected components
        max_dist = d[d < float('inf')].max() if np.any(d < float('inf')) else 1
        d[d == float('inf')] = max_dist + 1

        # Ideal distances scaled to canvas
        L0 = min(width, height) / (max_dist + 1) * 0.8
        L = d * L0

        # Spring constants: k_ij = 1 / d_ij²
        with np.errstate(divide='ignore'):
            K = 1 / (d * d)
        K[K == float('inf')] = 0
        np.fill_diagonal(K, 0)

        # Initialize positions randomly within canvas
        margin = 0.1
        pos = np.random.rand(n, 2) * [width * (1 - 2*margin), height * (1 - 2*margin)]
        pos += [width * margin, height * margin]

        # Iterative refinement
        epsilon = 1e-4
        for _ in range(iterations):
            # Find node with maximum partial derivative magnitude (delta)
            max_delta = 0
            max_node = 0

            for m in range(n):
                dx, dy = 0, 0
                for i in range(n):
                    if i == m:
                        continue
                    diff = pos[m] - pos[i]
                    dist = max(np.linalg.norm(diff), 0.01)
                    dx += K[m, i] * (diff[0] - L[m, i] * diff[0] / dist)
                    dy += K[m, i] * (diff[1] - L[m, i] * diff[1] / dist)

                delta = math.sqrt(dx*dx + dy*dy)
                if delta > max_delta:
                    max_delta = delta
                    max_node = m

            if max_delta < epsilon:
                break

            # Move the node with maximum delta using Newton-Raphson
            m = max_node
            for _ in range(5):  # Inner iterations
                dx, dy = 0, 0
                dxx, dyy, dxy = 0, 0, 0

                for i in range(n):
                    if i == m:
                        continue
                    diff = pos[m] - pos[i]
                    dist = max(np.linalg.norm(diff), 0.01)
                    dist3 = dist ** 3

                    # First derivatives
                    dx += K[m, i] * (diff[0] - L[m, i] * diff[0] / dist)
                    dy += K[m, i] * (diff[1] - L[m, i] * diff[1] / dist)

                    # Second derivatives (Hessian)
                    dxx += K[m, i] * (1 - L[m, i] * diff[1]**2 / dist3)
                    dyy += K[m, i] * (1 - L[m, i] * diff[0]**2 / dist3)
                    dxy += K[m, i] * L[m, i] * diff[0] * diff[1] / dist3

                # Solve 2x2 linear system: H × Δp = -∇E
                det = dxx * dyy - dxy * dxy
                if abs(det) > 1e-10:
                    pos[m, 0] -= (dyy * dx - dxy * dy) / det
                    pos[m, 1] -= (dxx * dy - dxy * dx) / det

        return {node: (pos[i, 0], pos[i, 1]) for i, node in enumerate(nodes)}


def layout_to_graphviz(
    positions: Dict[str, Tuple[float, float]],
    scale: float = 0.01
) -> Dict[str, str]:
    """
    Convert positions to Graphviz pos attributes.

    Args:
        positions: Node positions from layout algorithm
        scale: Scale factor (Graphviz uses inches)

    Returns:
        Dictionary mapping node IDs to pos strings like "1.5,2.3!"
    """
    return {
        node: f"{x * scale},{y * scale}!"
        for node, (x, y) in positions.items()
    }


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == '__main__':
    # Example: Simple MLP graph (like the image you showed)
    nodes = ['x', 'W', 'b', 'Wx', 'Wx+b', 'h', 'V', 'a', 'Vh', 'Vh+a', 'y']
    edges = [
        ('x', 'Wx'), ('W', 'Wx'),
        ('Wx', 'Wx+b'), ('b', 'Wx+b'),
        ('Wx+b', 'h'),
        ('h', 'Vh'), ('V', 'Vh'),
        ('Vh', 'Vh+a'), ('a', 'Vh+a'),
        ('Vh+a', 'y')
    ]

    print("Testing Force-Directed Layout Algorithms")
    print("=" * 50)

    # Test each algorithm
    algorithms = [
        ("Basic Spring Embedder", ForceDirectedLayout()),
        ("Fruchterman-Reingold", FruchtermanReingold()),
        ("Kamada-Kawai", KamadaKawai()),
    ]

    for name, algo in algorithms:
        positions = algo.layout(nodes, edges, iterations=100, seed=42)
        print(f"\n{name}:")
        for node, (x, y) in sorted(positions.items()):
            print(f"  {node:8s}: ({x:7.2f}, {y:7.2f})")

    # Compute energy for basic layout
    basic = ForceDirectedLayout()
    pos = basic.layout(nodes, edges, iterations=500, seed=42)
    energy = basic.compute_energy(nodes, edges, pos)
    print(f"\nFinal energy (Basic Spring): {energy:.2f}")
