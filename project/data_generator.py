"""
Data generation module for flow measurements.
Generates synthetic time series data with noise and patterns.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List
from .config import SimulationConfig


class FlowDataGenerator:
    """
    Generates synthetic flow measurement data.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the data generator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        np.random.seed(42)  # For reproducibility
    
    def generate_base_flow(self, num_samples: int, node_id: str) -> np.ndarray:
        """
        Generate base flow pattern with daily cycles.
        
        Args:
            num_samples: Number of samples to generate
            node_id: Identifier of the node
            
        Returns:
            Array of flow values
        """
        # Create time array
        t = np.arange(num_samples) * self.config.time_step_seconds
        
        # Daily cycle (24-hour period)
        daily_cycle = np.sin(2 * np.pi * t / (24 * 3600))
        
        # Weekly cycle (7-day period) - smaller amplitude
        weekly_cycle = 0.3 * np.sin(2 * np.pi * t / (7 * 24 * 3600))
        
        # Random variation based on node
        node_hash = hash(node_id) % 100
        node_factor = 0.8 + (node_hash / 100) * 0.4  # 0.8 to 1.2
        
        # Combine patterns
        base_flow = self.config.base_flow_rate * node_factor
        variation = self.config.base_flow_rate * self.config.flow_variation
        
        flow = base_flow + variation * (daily_cycle + weekly_cycle)
        
        # Ensure non-negative flows
        flow = np.maximum(flow, 0)
        
        return flow
    
    def add_noise(self, flow: np.ndarray) -> np.ndarray:
        """
        Add measurement noise to flow data.
        
        Args:
            flow: Clean flow values
            
        Returns:
            Noisy flow values
        """
        noise = np.random.normal(0, self.config.noise_std, len(flow))
        return flow + noise
    
    def generate_time_series(self, node_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Generate time series data for all nodes.
        
        Args:
            node_ids: List of node identifiers
            
        Returns:
            Dictionary mapping node IDs to DataFrames with time series
        """
        num_samples = self.config.total_samples
        
        # Generate time index
        time_index = pd.date_range(
            start=self.config.start_time,
            periods=num_samples,
            freq=f"{int(self.config.time_step_seconds)}s"
        )
        
        time_series = {}
        
        for node_id in node_ids:
            # Generate base flow
            base_flow = self.generate_base_flow(num_samples, node_id)
            
            # Add noise
            noisy_flow = self.add_noise(base_flow)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': time_index,
                'node_id': node_id,
                'flow': noisy_flow,
                'anomaly_type': 'none',
                'anomaly_active': False
            })
            
            time_series[node_id] = df
        
        return time_series
