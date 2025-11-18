"""
AI Graph Flow Simulator

A system for simulating measurement data in distribution networks
(e.g., water or gas pipelines) for flow balance analysis.
"""

from .anomaly_simulator import AnomalySimulator
from .config import SimulationConfig
from .data_generator import FlowDataGenerator
from .simulator import FlowSimulator
from .network_topology import NetworkTopology
from .utils import SimulationDataSaver
from .visualizer import FlowVisualizer

__version__ = "0.1.0"

__all__ = [
    'SimulationConfig',
    'NetworkTopology',
    'FlowDataGenerator',
    'AnomalySimulator',
    'FlowVisualizer',
    'FlowSimulator',
    'SimulationDataSaver',
]
