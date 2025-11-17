"""
Configuration module for simulation parameters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the simulation.
    """
    # Network topology
    num_nodes: int = 20
    num_sources: int = 1
    topology_type: str = "tree"  # tree, mesh, random
    edge_length_range: Tuple[float, float] = (5.0, 25.0)
    source_nodes: Optional[List[str]] = None
    
    # Time parameters
    start_time: datetime = None
    duration_hours: int = 24
    sampling_frequency_hz: float = 1.0  # 1 Hz = 1 sample/second
    
    # Flow parameters
    base_flow_rate: float = 100.0  # Base flow rate in m³/h
    flow_variation: float = 0.2  # 20% variation
    noise_std: float = 2.0  # Standard deviation of measurement noise
    
    # Anomaly parameters
    anomaly_probability: float = 0.1  # Probability of anomaly occurrence
    progressive_leak_probability: float = 0.25
    leak_magnitude_range: tuple = (5.0, 15.0)  # Flow loss in m³/h
    meter_error_range: tuple = (-5.0, 5.0)  # Meter offset in m³/h
    
    # Output
    output_dir: str = "output"
    export_format: str = "csv"  # csv or json
    real_time_mode: bool = False
    stream_buffer_size: int = 2048
    database_url: Optional[str] = None  # Timescale/Influx connection string
    auth_public_key: Optional[str] = None  # JWT verification key
    jwt_audience: Optional[str] = None
    dashboard_enabled: bool = True
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8080
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.start_time is None:
            self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if not self.source_nodes:
            self.source_nodes = [f"source_{idx+1:02d}" for idx in range(self.num_sources)]
        else:
            self.num_sources = len(self.source_nodes)
    
    @property
    def end_time(self) -> datetime:
        """Calculate end time based on duration."""
        return self.start_time + timedelta(hours=self.duration_hours)
    
    @property
    def total_samples(self) -> int:
        """Calculate total number of samples."""
        total_seconds = self.duration_hours * 3600
        return int(total_seconds * self.sampling_frequency_hz)
    
    @property
    def time_step_seconds(self) -> float:
        """Get time step in seconds between samples."""
        return 1.0 / self.sampling_frequency_hz
