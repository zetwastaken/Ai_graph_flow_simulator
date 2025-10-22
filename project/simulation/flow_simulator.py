# -*- coding: utf-8 -*-
"""
Simulates data flow through the network graph.
"""
import numpy as np
import random

class FlowSimulator:
    """
    Generates synthetic time-series data for flows in the network.
    """
    def __init__(self, graph, config):
        """
        Initializes the FlowSimulator.

        Args:
            graph (nx.DiGraph): The network topology.
            config (dict): Simulation parameters.
        """
        self.graph = graph
        self.config = config
        self.node_states = {node: {"base_flow": random.uniform(10, 100)} for node in graph.nodes()}

    def simulate_step(self, time_step, anomaly_injector):
        """
        Simulates one step of data generation for all nodes.

        Args:
            time_step (int): The current time step of the simulation.
            anomaly_injector (AnomalyInjector): The anomaly injector instance.

        Returns:
            list: A list of data points for the current time step.
        """
        step_data = []
        
        # Base flow for sources
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'source':
                base_flow = self.node_states[node]["base_flow"]
                # Add some cyclical pattern
                flow = base_flow * (1 + 0.2 * np.sin(2 * np.pi * time_step / (24 * 60))) # Daily cycle
                # Add noise
                flow *= (1 + np.random.normal(0, self.config['noise_level']))
                
                data_point = {
                    "timestamp": time_step,
                    "node": node,
                    "flow": flow,
                    "anomaly": "none"
                }
                step_data.append(data_point)

        # Propagate flow to other nodes
        for node, data in self.graph.nodes(data=True):
             if data.get('type') != 'source':
                in_flow = sum(step_data[p]['flow'] for p in self.graph.predecessors(node) if p < len(step_data))
                
                # Apply leaks on edges
                for pred in self.graph.predecessors(node):
                    for anomaly in anomaly_injector.active_anomalies:
                        if anomaly['type'] == 'leak' and anomaly['edge'] == (pred, node):
                            in_flow *= (1 - anomaly['magnitude'])

                # Add noise
                flow = in_flow * (1 + np.random.normal(0, self.config['noise_level']))

                data_point = {
                    "timestamp": time_step,
                    "node": node,
                    "flow": flow,
                    "anomaly": "none"
                }
                step_data.append(data_point)

        # Apply meter anomalies
        final_step_data = []
        for dp in step_data:
            final_dp = anomaly_injector.apply_anomalies(dp.copy(), time_step)
            final_step_data.append(final_dp)

        return final_step_data
