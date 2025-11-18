"""Data generation module for node and edge flows."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import SimulationConfig
from .network_topology import NetworkTopology


class FlowDataGenerator:
    """
    Generates synthetic flow measurement data.
    """

    def __init__(self, config: SimulationConfig, topology: Optional[NetworkTopology] = None):
        """
        Initialize the flow data generator.

        Args:
            config: Simulation configuration
            topology: Network topology (optional, can be attached later)
        """
        self.config = config
        self.topology = topology
        if hasattr(self.config, "seed") and self.config.seed is not None:
            np.random.seed(self.config.seed)

    def attach_topology(self, topology: NetworkTopology):
        """
        Attach a network topology to the generator.

        Args:
            topology: Network topology instance
        """
        self.topology = topology

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

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def _ensure_topology(self, topology: Optional[NetworkTopology]):
        if topology is not None:
            self.topology = topology
        if self.topology is None:
            raise ValueError("Topology must be provided before generating data")

    def _time_index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.config.start_time,
            periods=self.config.total_samples,
            freq=f"{int(max(1, self.config.time_step_seconds))}s",
        )

    def _generate_consumer_series(self, node_ids: List[str], time_index: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        series: Dict[str, pd.DataFrame] = {}
        num_samples = self.config.total_samples
        for node_id in node_ids:
            base_flow = self.generate_base_flow(num_samples, node_id)
            noisy_flow = self.add_noise(base_flow)
            df = pd.DataFrame(
                {
                    "timestamp": time_index,
                    "node_id": node_id,
                    "flow": noisy_flow,
                    "node_type": "consumer",
                    "anomaly_type": "none",
                    "anomaly_active": False,
                }
            )
            series[node_id] = df
        return series

    def _propagate_internal_nodes(
        self, consumer_series: Dict[str, pd.DataFrame], time_index: pd.DatetimeIndex
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
        graph = self.topology.graph  # type: ignore[union-attr]
        node_flow_arrays: Dict[str, np.ndarray] = {}
        node_series: Dict[str, pd.DataFrame] = dict(consumer_series)
        for node_id, df in consumer_series.items():
            node_flow_arrays[node_id] = df["flow"].to_numpy()

        try:
            order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            order = list(graph.nodes())

        for node in reversed(order):
            if node in node_flow_arrays:
                continue
            children = list(graph.successors(node))
            flow = np.zeros(self.config.total_samples)
            if children:
                for child in children:
                    flow += node_flow_arrays.get(child, np.zeros(self.config.total_samples))
            df = pd.DataFrame(
                {
                    "timestamp": time_index,
                    "node_id": node,
                    "flow": flow,
                    "node_type": self.topology.get_node_type(node),
                    "anomaly_type": "none",
                    "anomaly_active": False,
                }
            )
            node_series[node] = df
            node_flow_arrays[node] = flow

        return node_series, node_flow_arrays

    def _build_edge_series(
        self, node_flows: Dict[str, np.ndarray], time_index: pd.DatetimeIndex
    ) -> Dict[str, pd.DataFrame]:
        edge_series: Dict[str, pd.DataFrame] = {}
        for source, target, data in self.topology.graph.edges(data=True):  # type: ignore[union-attr]
            edge_id = data.get("edge_id", f"e_{source}_{target}")
            flow = node_flows.get(target, np.zeros(self.config.total_samples))
            df = pd.DataFrame(
                {
                    "timestamp": time_index,
                    "edge_id": edge_id,
                    "source": source,
                    "target": target,
                    "length": data.get('length'),
                    "flow": flow,
                    "anomaly_type": "none",
                    "anomaly_active": False,
                }
            )
            edge_series[edge_id] = df
        return edge_series

    def generate_time_series(
        self, topology: Optional[NetworkTopology] = None, node_ids: Optional[List[str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Return both node and edge time series for the provided topology."""

        self._ensure_topology(topology)
        time_index = self._time_index()
        topology = self.topology  # type: ignore[assignment]
        node_ids = node_ids or topology.get_consumers()
        consumer_series = self._generate_consumer_series(node_ids, time_index)
        node_series, node_flow_arrays = self._propagate_internal_nodes(consumer_series, time_index)
        edge_series = self._build_edge_series(node_flow_arrays, time_index)
        return node_series, edge_series
