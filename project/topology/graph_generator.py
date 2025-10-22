# -*- coding: utf-8 -*-
"""
Generates the network topology as a graph.
"""
import networkx as nx
import random

class GraphGenerator:
    """
    Generates the network topology as a graph.
    Nodes represent measurement points, and edges represent connections.
    """
    def __init__(self, num_nodes, num_sources):
        """
        Initializes the graph generator.

        Args:
            num_nodes (int): The total number of nodes in the graph.
            num_sources (int): The number of source nodes.
        """
        self.num_nodes = num_nodes
        self.num_sources = num_sources
        self.graph = nx.DiGraph()

    def generate_topology(self):
        """
        Creates a random graph representing the network topology.
        Source nodes are randomly selected and have 'source' attribute.
        """
        if self.num_nodes <= 0:
            return

        # Add nodes
        for i in range(self.num_nodes):
            self.graph.add_node(i, type='meter')

        # Select sources
        source_nodes = random.sample(range(self.num_nodes), self.num_sources)
        for node in source_nodes:
            self.graph.nodes[node]['type'] = 'source'

        # Add edges to create a connected graph
        for i in range(self.num_nodes):
            if i not in source_nodes:
                # Connect non-source nodes to other random nodes
                num_connections = random.randint(1, 3)
                for _ in range(num_connections):
                    target = random.randint(0, self.num_nodes - 1)
                    if i != target:
                        self.graph.add_edge(target, i, weight=random.uniform(0.5, 2.0))

        # Ensure all nodes are reachable from a source
        for node in self.graph.nodes():
            if not any(nx.has_path(self.graph, source, node) for source in source_nodes):
                # Connect to a random source if not reachable
                source_to_connect = random.choice(source_nodes)
                self.graph.add_edge(source_to_connect, node, weight=random.uniform(0.5, 2.0))
        
        print("Topology generated.")
        return self.graph

    def get_graph(self):
        """
        Returns the generated graph.
        """
        return self.graph
