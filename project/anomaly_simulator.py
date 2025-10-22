"""
Anomaly simulation module.
Simulates leaks and meter errors in the network.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
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
    
    def generate_anomalies(self, node_ids: List[str], edge_ids: List[str]) -> List[Dict]:
        """
        Generate random anomalies for the simulation.
        
        Args:
            node_ids: List of node identifiers
            edge_ids: List of edge identifiers
            
        Returns:
            List of anomaly definitions
        """
        anomalies = []
        
        # Calculate number of anomalies to generate
        total_duration_minutes = self.config.duration_hours * 60
        num_anomalies = int(total_duration_minutes * self.config.anomaly_probability / 60)
        
        for i in range(num_anomalies):
            # Random start time
            start_offset = np.random.randint(0, self.config.duration_hours * 60)
            start_time = self.config.start_time + timedelta(minutes=start_offset)
            
            # Random duration (10 to 120 minutes)
            duration_minutes = np.random.randint(10, 120)
            
            # Random anomaly type
            anomaly_type = np.random.choice(['leak', 'meter_error'])
            
            if anomaly_type == 'leak':
                # Leak on an edge
                target_edge = np.random.choice(edge_ids)
                magnitude = np.random.uniform(*self.config.leak_magnitude_range)
                
                anomaly = {
                    'id': f'anom_{i+1:03d}',
                    'type': 'leak',
                    'start_time': start_time,
                    'duration_minutes': duration_minutes,
                    'target_type': 'edge',
                    'target_id': target_edge,
                    'magnitude': magnitude,
                    'mode': 'const'
                }
            else:
                # Meter error on a node
                target_node = np.random.choice(node_ids)
                magnitude = np.random.uniform(*self.config.meter_error_range)
                mode = np.random.choice(['add', 'mul', 'drift'])
                
                # Adjust magnitude for multiplicative errors
                if mode == 'mul':
                    magnitude = np.random.uniform(0.8, 1.2)
                
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
    
    def apply_anomalies(self, time_series: Dict[str, pd.DataFrame], 
                       edge_flows: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
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
            
            elif anomaly['type'] == 'leak' and edge_flows is not None:
                # Apply to edge flows
                target_id = anomaly['target_id']
                if target_id in edge_flows:
                    df = edge_flows[target_id]
                    mask = (df['timestamp'] >= anomaly['start_time']) & (df['timestamp'] < end_time)
                    df.loc[mask, 'flow'] -= anomaly['magnitude']
                    df.loc[mask, 'flow'] = df.loc[mask, 'flow'].clip(lower=0)
                    df.loc[mask, 'anomaly_type'] = anomaly['type']
                    df.loc[mask, 'anomaly_active'] = True
        
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
