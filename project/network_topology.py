"""Configurable network topology generation utilities."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import networkx as nx

from .config import SimulationConfig


class NetworkTopology:
    """Create directed network topologies for the simulator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.graph = nx.DiGraph()
        self._create_topology()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def get_edges(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())

    def get_edge_id(self, source: str, target: str) -> str:
        return self.graph.edges[source, target].get("edge_id", f"e_{source}_{target}")

    def get_node_type(self, node: str) -> str:
        return self.graph.nodes[node].get("node_type", "unknown")

    def get_consumers(self) -> List[str]:
        return [n for n in self.graph.nodes() if self.get_node_type(n) == "consumer"]

    def get_topology_info(self) -> Dict:
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
