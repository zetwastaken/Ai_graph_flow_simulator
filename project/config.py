"""
Configuration module for simulation parameters.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the simulation.
    """
    # Network topology
    num_nodes: int = 20
    num_sources: int = 1
    
    # Time parameters
    start_time: datetime = None
    duration_hours: int = 24
    sampling_frequency_hz: float = 1.0  # 1 Hz = 1 sample per second
    
    # Flow parameters
    base_flow_rate: float = 100.0  # Base flow rate in m³/h
    flow_variation: float = 0.2  # 20% variation
    noise_std: float = 2.0  # Standard deviation of measurement noise
    
    # Anomaly parameters
    anomaly_probability: float = 0.1  # Probability of anomaly occurrence
    leak_magnitude_range: tuple = (5.0, 15.0)  # Flow loss in m³/h
    meter_error_range: tuple = (-5.0, 5.0)  # Meter offset in m³/h
    
    # Output
    output_dir: str = "output"
    export_format: str = "csv"  # csv or json
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.start_time is None:
            self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
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
