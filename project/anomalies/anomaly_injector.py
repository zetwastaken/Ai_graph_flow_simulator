# -*- coding: utf-8 -*-
"""
Injects anomalies into the simulation data.
"""
import random
import numpy as np

class AnomalyInjector:
    """
    Injects anomalies into the flow data.
    """
    def __init__(self, anomaly_config):
        """
        Initializes the AnomalyInjector.

        Args:
            anomaly_config (dict): Configuration for anomalies.
        """
        self.anomaly_config = anomaly_config
        self.active_anomalies = []

    def introduce_anomaly(self, graph, current_time):
        """
        Randomly introduces an anomaly into the network.
        """
        anomaly_type = random.choice(list(self.anomaly_config.keys()))
        
        if anomaly_type == "leak":
            # Choose a random edge for the leak
            edge = random.choice(list(graph.edges()))
            leak_magnitude = random.uniform(0.1, 0.5) # 10-50% leak
            self.active_anomalies.append({
                "type": "leak",
                "edge": edge,
                "magnitude": leak_magnitude,
                "start_time": current_time
            })
            print(f"Leak injected at edge {edge} at time {current_time}")

        elif anomaly_type == "meter_error":
            # Choose a random node for the meter error
            node = random.choice(list(graph.nodes()))
            error_type = self.anomaly_config["meter_error"]["type"]
            offset = random.uniform(5, 20) if error_type == "offset" else 0
            drift = random.uniform(0.01, 0.05) if error_type == "drift" else 0
            self.active_anomalies.append({
                "type": "meter_error",
                "node": node,
                "offset": offset,
                "drift": drift,
                "start_time": current_time
            })
            print(f"Meter error injected at node {node} at time {current_time}")

    def apply_anomalies(self, data_point, current_time):
        """
        Applies active anomalies to a data point.
        """
        data_point['anomaly'] = 'none'
        for anomaly in self.active_anomalies:
            if anomaly['type'] == 'leak':
                # Leaks are handled in the flow calculation, just mark the data
                if data_point['node'] in anomaly['edge']:
                     data_point['anomaly'] = 'leak'
            
            elif anomaly['type'] == 'meter_error' and data_point['node'] == anomaly['node']:
                data_point['anomaly'] = 'meter_error'
                if "offset" in anomaly:
                    data_point['flow'] += anomaly['offset']
                if "drift" in anomaly:
                    time_delta = current_time - anomaly['start_time']
                    data_point['flow'] += anomaly['drift'] * time_delta
        return data_point
