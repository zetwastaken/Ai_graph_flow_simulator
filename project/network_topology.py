"""
Network topology generation module.
Creates virtual network topology as a graph structure.
"""

import networkx as nx
from typing import Dict, List, Tuple


class NetworkTopology:
    """
    Represents a distribution network topology.
    Nodes represent measurement points, edges represent connections.
    """
    
    def __init__(self, num_nodes: int = 10, num_sources: int = 1):
        """
        Initialize network topology.
        
        Args:
            num_nodes: Number of consumption nodes
            num_sources: Number of supply sources
        """
        self.num_nodes = num_nodes
        self.num_sources = num_sources
        self.graph = nx.DiGraph()
        self._create_topology()
    
    def _create_topology(self):
        """Create network topology as a directed graph."""
        # Add root/source node
        self.graph.add_node("root", node_type="source", demand=0)
        
        # Create hub nodes (main distribution points)
        hubs = ["hub_north", "hub_south", "hub_east", "hub_west"][:min(4, max(1, self.num_nodes // 5))]
        for hub in hubs:
            self.graph.add_node(hub, node_type="hub", demand=0)
            edge_id = f"e_root_{hub.split('_')[1]}"
            self.graph.add_edge("root", hub, edge_id=edge_id, length=10.0)
        
        # Create consumer nodes distributed among hubs
        nodes_per_hub = max(1, (self.num_nodes - len(hubs)) // len(hubs))
        node_counter = 1
        
        for hub in hubs:
            for i in range(nodes_per_hub):
                if node_counter > self.num_nodes:
                    break
                node_id = f"c{node_counter:02d}"
                self.graph.add_node(node_id, node_type="consumer", demand=10.0)
                edge_id = f"e_{hub}_{node_id}"
                self.graph.add_edge(hub, node_id, edge_id=edge_id, length=5.0)
                node_counter += 1
    
    def get_nodes(self) -> List[str]:
        """Get all nodes in the network."""
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the network."""
        return list(self.graph.edges())
    
    def get_edge_id(self, source: str, target: str) -> str:
        """Get edge ID for a connection."""
        return self.graph.edges[source, target].get("edge_id", f"e_{source}_{target}")
    
    def get_node_type(self, node: str) -> str:
        """Get type of a node."""
        return self.graph.nodes[node].get("node_type", "unknown")
    
    def get_consumers(self) -> List[str]:
        """Get all consumer nodes."""
        return [n for n in self.graph.nodes() if self.get_node_type(n) == "consumer"]
    
    def get_topology_info(self) -> Dict:
        """Get summary information about the topology."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_sources": sum(1 for n in self.graph.nodes() if self.get_node_type(n) == "source"),
            "num_consumers": len(self.get_consumers()),
            "num_hubs": sum(1 for n in self.graph.nodes() if self.get_node_type(n) == "hub")
        }
