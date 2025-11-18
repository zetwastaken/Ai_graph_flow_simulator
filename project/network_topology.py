"""Configurable network topology generation utilities."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import networkx as nx

from .config import SimulationConfig


class NetworkTopology:
    """Create directed network topologies for the simulator."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize network topology.

        Args:
            config: Simulation configuration containing topology parameters
        """
        self.config = config
        self.graph = nx.DiGraph()
        self._create_topology()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_nodes(self) -> List[str]:
        """
        Get all node IDs in the network.

        Returns:
            List of node identifiers
        """
        return list(self.graph.nodes())

    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Get all edges in the network.

        Returns:
            List of (source, target) tuples
        """
        return list(self.graph.edges())

    def get_edge_id(self, source: str, target: str) -> str:
        """
        Get the unique identifier for an edge.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Edge identifier string
        """
        return self.graph.edges[source, target].get("edge_id", f"e_{source}_{target}")

    def get_node_type(self, node: str) -> str:
        """
        Get the type of a node.

        Args:
            node: Node identifier

        Returns:
            Node type (source, hub, consumer, or unknown)
        """
        return self.graph.nodes[node].get("node_type", "unknown")

    def get_consumers(self) -> List[str]:
        """
        Get all consumer nodes in the network.

        Returns:
            List of consumer node identifiers
        """
        return [n for n in self.graph.nodes() if self.get_node_type(n) == "consumer"]

    def get_topology_info(self) -> Dict:
        """
        Get summary information about the network topology.

        Returns:
            Dictionary containing topology statistics
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_sources": sum(1 for n in self.graph.nodes() if self.get_node_type(n) == "source"),
            "num_consumers": len(self.get_consumers()),
            "num_hubs": sum(1 for n in self.graph.nodes() if self.get_node_type(n) == "hub"),
            "topology_type": self.config.topology_type,
        }

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _edge_length(self) -> float:
        low, high = self.config.edge_length_range
        return round(random.uniform(low, high), 2)

    def _register_node(self, node_id: str, node_type: str, **attrs):
        attrs.setdefault("demand", 0.0)
        self.graph.add_node(node_id, node_type=node_type, **attrs)

    def _add_edge(self, source: str, target: str):
        edge_id = f"e_{source}_{target}"
        self.graph.add_edge(source, target, edge_id=edge_id, length=self._edge_length())

    def _create_topology(self):
        builders = {
            "tree": self._build_tree_topology,
            "mesh": self._build_mesh_topology,
            "random": self._build_random_topology,
            "radial": self._build_radial_topology,
            "grid": self._build_grid_topology,
        }
        builders.get(self.config.topology_type, self._build_tree_topology)()

    def _build_tree_topology(self):
        self.graph.clear()
        for source in self.config.source_nodes:
            self._register_node(source, "source")

        hub_count = max(1, self.config.num_nodes // 5)
        hubs = [f"hub_{idx+1:02d}" for idx in range(hub_count)]
        for idx, hub in enumerate(hubs):
            self._register_node(hub, "hub")
            source = self.config.source_nodes[idx % len(self.config.source_nodes)]
            self._add_edge(source, hub)

        consumer_idx = 1
        while consumer_idx <= self.config.num_nodes:
            for hub in hubs:
                if consumer_idx > self.config.num_nodes:
                    break
                consumer = f"c{consumer_idx:03d}"
                self._register_node(consumer, "consumer", demand=10.0)
                self._add_edge(hub, consumer)
                consumer_idx += 1

    def _build_mesh_topology(self):
        self._build_tree_topology()
        hubs = [n for n in self.graph.nodes() if self.get_node_type(n) == "hub"]
        if len(hubs) > 1:
            for i, hub in enumerate(hubs[:-1]):
                nxt = hubs[i + 1]
                if not self.graph.has_edge(hub, nxt):
                    self._add_edge(hub, nxt)
        consumers = self.get_consumers()
        for consumer in consumers:
            alt_parent = random.choice(hubs)
            if not self.graph.has_edge(alt_parent, consumer):
                self._add_edge(alt_parent, consumer)

    def _build_random_topology(self):
        self.graph.clear()
        all_nodes: List[str] = []
        for source in self.config.source_nodes:
            self._register_node(source, "source")
            all_nodes.append(source)
        consumers = [f"c{idx+1:03d}" for idx in range(self.config.num_nodes)]
        hubs = random.sample(consumers, k=max(1, self.config.num_nodes // 6))
        for consumer in consumers:
            node_type = "hub" if consumer in hubs else "consumer"
            self._register_node(consumer, node_type, demand=10.0)
            all_nodes.append(consumer)

        # Ensure each non-source has at least one parent
        ordered_sources = list(self.config.source_nodes)
        for node in consumers:
            parent_pool = ordered_sources + [h for h in hubs if h != node]
            random.shuffle(parent_pool)
            parent_pool = [p for p in parent_pool if p != node]
            parent_count = random.randint(1, min(3, len(parent_pool)))
            for parent in parent_pool[:parent_count]:
                self._add_edge(parent, node)

        # Add extra random edges to increase redundancy
        for _ in range(self.config.num_nodes):
            src, dst = sorted(random.sample(consumers, 2))
            if src != dst:
                self._add_edge(src, dst)

    def _build_radial_topology(self):
        self.graph.clear()
        sources = list(self.config.source_nodes)
        for source in sources:
            self._register_node(source, "source")

        hub_count = max(len(sources) * 2, 4)
        hubs = [f"ring_{idx+1:02d}" for idx in range(hub_count)]
        for hub in hubs:
            self._register_node(hub, "hub")
        for idx, hub in enumerate(hubs):
            source = sources[idx % len(sources)]
            self._add_edge(source, hub)

        consumers = [f"c{idx+1:03d}" for idx in range(self.config.num_nodes)]
        node_iter = iter(consumers)
        remaining = list(consumers)
        for hub in hubs:
            for _ in range(max(1, self.config.num_nodes // hub_count)):
                if not remaining:
                    break
                consumer = remaining.pop(0)
                self._register_node(consumer, "consumer", demand=10.0)
                self._add_edge(hub, consumer)
        for consumer in remaining:
            hub = random.choice(hubs)
            self._register_node(consumer, "consumer", demand=10.0)
            self._add_edge(hub, consumer)

        # Connect hubs in ring for redundancy
        for idx, hub in enumerate(hubs):
            nxt = hubs[(idx + 1) % len(hubs)]
            self._add_edge(hub, nxt)

    def _build_grid_topology(self):
        self.graph.clear()
        sources = list(self.config.source_nodes)
        for source in sources:
            self._register_node(source, "source")

        grid_size = max(2, math.ceil(math.sqrt(self.config.num_nodes)))
        consumers: List[str] = []
        for r in range(grid_size):
            for c in range(grid_size):
                if len(consumers) >= self.config.num_nodes:
                    break
                node_id = f"g{r:02d}_{c:02d}"
                consumers.append(node_id)
                self._register_node(node_id, "consumer", demand=10.0)
            if len(consumers) >= self.config.num_nodes:
                break

        # Connect grid neighbors (right and down) to form mesh
        for r in range(grid_size):
            for c in range(grid_size):
                node_id = f"g{r:02d}_{c:02d}"
                if node_id not in consumers:
                    continue
                if c + 1 < grid_size:
                    neighbor = f"g{r:02d}_{c+1:02d}"
                    if neighbor in consumers:
                        self._add_edge(node_id, neighbor)
                if r + 1 < grid_size:
                    neighbor = f"g{r+1:02d}_{c:02d}"
                    if neighbor in consumers:
                        self._add_edge(node_id, neighbor)

        # Attach sources evenly across top row
        top_row = [f"g00_{c:02d}" for c in range(grid_size) if f"g00_{c:02d}" in consumers]
        for idx, node in enumerate(top_row):
            source = sources[idx % len(sources)]
            self._add_edge(source, node)
