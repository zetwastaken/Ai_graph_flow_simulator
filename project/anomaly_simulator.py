"""
Anomaly simulation module.
Simulates leaks and meter errors in the network.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import networkx as nx

from .config import SimulationConfig


class AnomalySimulator:
    """
    Simulates anomalies in flow measurements.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the anomaly simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.anomalies = []
        self.graph: Optional[nx.DiGraph] = None
        self.edge_catalog: Dict[str, Tuple[str, str]] = {}

    def attach_topology(self, graph: nx.DiGraph):
        """Store reference graph for cascade calculations."""
        self.graph = graph
    
    def generate_anomalies(
        self, node_ids: List[str], edge_catalog: Dict[str, Tuple[str, str]]
    ) -> List[Dict]:
        """
        Generate random anomalies for the simulation.
        
        Args:
            node_ids: List of node identifiers
            edge_ids: List of edge identifiers
            
        Returns:
            List of anomaly definitions
        """
        self.edge_catalog = edge_catalog
        anomalies = []
        anomaly_types = []
        if self.config.enable_leaks:
            anomaly_types.append('leak')
        if self.config.enable_meter_errors:
            anomaly_types.append('meter_error')
        if not anomaly_types:
            self.anomalies = []
            return []
        
        severity = max(0.1, self.config.anomaly_severity)
        leak_range = tuple(val * severity for val in self.config.leak_magnitude_range)
        meter_range = tuple(val * severity for val in self.config.meter_error_range)
        # Calculate number of anomalies to generate
        total_duration_minutes = self.config.duration_hours * 60
        num_anomalies = int(total_duration_minutes * self.config.anomaly_probability *
                            self.config.anomaly_rate_multiplier / 60)
        
        for i in range(num_anomalies):
            # Random start time
            start_offset = np.random.randint(0, self.config.duration_hours * 60)
            start_time = self.config.start_time + timedelta(minutes=start_offset)
            
            # Random duration (10 to 120 minutes)
            duration_minutes = np.random.randint(10, 120)
            
            # Random anomaly type constrained by config
            anomaly_type = np.random.choice(anomaly_types)
            
            if anomaly_type == 'leak':
                target_edge = np.random.choice(list(edge_catalog.keys()))
                magnitude = np.random.uniform(*leak_range)
                progressive = np.random.rand() < self.config.progressive_leak_probability
                anomaly = {
                    'id': f'anom_{i+1:03d}',
                    'type': 'leak',
                    'start_time': start_time,
                    'duration_minutes': duration_minutes,
                    'target_type': 'edge',
                    'target_id': target_edge,
                    'magnitude': magnitude,
                    'mode': 'progressive' if progressive else 'const'
                }
            else:
                # Meter error on a node
                target_node = np.random.choice(node_ids)
                magnitude = np.random.uniform(*meter_range)
                mode = np.random.choice(['add', 'mul', 'drift'])
                
                # Adjust magnitude for multiplicative errors
                if mode == 'mul':
                    factor = np.random.uniform(0.8, 1.2)
                    magnitude = 1 + (factor - 1) * severity
                
                anomaly = {
                    'id': f'anom_{i+1:03d}',
                    'type': 'meter_error',
                    'start_time': start_time,
                    'duration_minutes': duration_minutes,
                    'target_type': 'node',
                    'target_id': target_node,
                    'magnitude': magnitude,
                    'mode': mode
                }
            
            anomalies.append(anomaly)
        
        self.anomalies = anomalies
        return anomalies
    
    def _leak_profile(self, samples: int, anomaly: Dict) -> np.ndarray:
        if samples <= 0:
            return np.array([])
        magnitude = anomaly['magnitude']
        if anomaly.get('mode') == 'progressive':
            return np.linspace(0, magnitude, samples)
        return np.full(samples, magnitude)

    def _propagate_leak(self, node_id: str, start: datetime, end: datetime,
                        profile: np.ndarray, node_series: Dict[str, pd.DataFrame]):
        if node_id not in node_series:
            return
        df = node_series[node_id]
        mask = (df['timestamp'] >= start) & (df['timestamp'] < end)
        samples = mask.sum()
        if samples == 0:
            return
        reduction = profile
        if len(profile) != samples and samples > 0:
            reduction = np.interp(
                np.linspace(0, len(profile) - 1, samples),
                np.arange(len(profile)),
                profile,
            )
        df.loc[mask, 'flow'] = (df.loc[mask, 'flow'] - reduction).clip(lower=0)
        df.loc[mask, 'anomaly_type'] = 'leak'
        df.loc[mask, 'anomaly_active'] = True
        if self.graph is None:
            return
        for child in self.graph.successors(node_id):
            self._propagate_leak(child, start, end, reduction, node_series)

    def apply_anomalies(self, time_series: Dict[str, pd.DataFrame], 
                       edge_flows: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply anomalies to the time series data.
        
        Args:
            time_series: Dictionary of time series DataFrames
            edge_flows: Dictionary of edge flow DataFrames (optional)
            
        Returns:
            Modified time series with anomalies applied
        """
        for anomaly in self.anomalies:
            end_time = anomaly['start_time'] + timedelta(minutes=anomaly['duration_minutes'])
            
            if anomaly['type'] == 'meter_error':
                # Apply to node measurements
                target_id = anomaly['target_id']
                if target_id in time_series:
                    df = time_series[target_id]
                    mask = (df['timestamp'] >= anomaly['start_time']) & (df['timestamp'] < end_time)
                    
                    if anomaly['mode'] == 'add':
                        df.loc[mask, 'flow'] += anomaly['magnitude']
                    elif anomaly['mode'] == 'mul':
                        df.loc[mask, 'flow'] *= anomaly['magnitude']
                    elif anomaly['mode'] == 'drift':
                        # Linear drift over time
                        drift_indices = np.where(mask)[0]
                        if len(drift_indices) > 0:
                            drift_values = np.linspace(0, anomaly['magnitude'], len(drift_indices))
                            df.loc[mask, 'flow'] += drift_values
                    
                    df.loc[mask, 'anomaly_type'] = anomaly['type']
                    df.loc[mask, 'anomaly_active'] = True
            
            elif anomaly['type'] == 'leak':
                target_id = anomaly['target_id']
                if target_id in edge_flows:
                    df = edge_flows[target_id]
                    mask = (df['timestamp'] >= anomaly['start_time']) & (df['timestamp'] < end_time)
                    samples = mask.sum()
                    if samples == 0:
                        continue
                    profile = self._leak_profile(samples, anomaly)
                    df.loc[mask, 'flow'] = (df.loc[mask, 'flow'] - profile).clip(lower=0)
                    df.loc[mask, 'anomaly_type'] = anomaly['type']
                    df.loc[mask, 'anomaly_active'] = True
                    if self.edge_catalog:
                        _, target_node = self.edge_catalog[target_id]
                        self._propagate_leak(target_node, anomaly['start_time'], end_time, profile, time_series)
        
        return time_series
    
    def get_anomaly_report(self) -> pd.DataFrame:
        """
        Get a report of all generated anomalies.
        
        Returns:
            DataFrame with anomaly information
        """
        if not self.anomalies:
            return pd.DataFrame()
        
        return pd.DataFrame(self.anomalies)
