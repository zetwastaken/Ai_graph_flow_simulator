"""
AI Graph Flow Simulator

A system for simulating measurement data in distribution networks
(e.g., water or gas pipelines) for flow balance analysis.
"""

from .config import SimulationConfig
from .network_topology import NetworkTopology
from .data_generator import FlowDataGenerator
from .anomaly_simulator import AnomalySimulator
from .visualizer import FlowVisualizer
from .simulator import FlowSimulator
from .streaming.streaming_simulator import StreamingSimulator
from .api.server import app as api_app

__version__ = "0.1.0"

__all__ = [
    'SimulationConfig',
    'NetworkTopology',
    'FlowDataGenerator',
    'AnomalySimulator',
    'FlowVisualizer',
    'FlowSimulator',
    'StreamingSimulator',
    'api_app'
]
