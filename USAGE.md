# AI Graph Flow Simulator - Usage Guide

## Quick Start

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulator

Run the default simulation:
```bash
python main.py
```

This will:
- Create a network topology with 20 nodes
- Generate 24 hours of flow data
- Simulate random anomalies (leaks and meter errors)
- Save data to the `output/` directory
- Generate visualization plots
- Create a summary report

## Output Files

The simulator generates the following files in the `output/` directory:

- `flow_measurements.csv` - Time series flow data for all nodes
- `anomalies.csv` - List of all generated anomalies
- `topology_info.json` - Network topology information
- `simulation_report.json` - Complete simulation summary
- `flow_plot.png` - Flow visualization for sample nodes
- `flow_statistics.png` - Statistical analysis of flows
- `anomaly_distribution.png` - Anomaly type and magnitude distribution

## Custom Configuration

You can customize the simulation by modifying the configuration in `main.py`:

```python
from project import FlowSimulator, SimulationConfig
from datetime import datetime

config = SimulationConfig(
    num_nodes=50,                    # Number of consumer nodes
    num_sources=1,                   # Number of supply sources
    duration_hours=48,               # Simulation duration
    sampling_frequency_hz=1.0,       # Sampling rate (Hz)
    base_flow_rate=100.0,           # Base flow in m³/h
    flow_variation=0.2,             # Flow variation (20%)
    noise_std=2.0,                  # Measurement noise std dev
    anomaly_probability=0.1,        # Anomaly occurrence probability
    output_dir="my_output",         # Output directory
    export_format="json"            # Export format (csv or json)
)

simulator = FlowSimulator(config)
simulator.setup()
simulator.run()
simulator.save_data()
simulator.visualize()
simulator.print_summary()
```

## Module Overview

### Network Topology (`network_topology.py`)
Creates a directed graph representing the distribution network with source nodes, hubs, and consumer nodes.

### Data Generator (`data_generator.py`)
Generates synthetic time series flow data with daily/weekly patterns and measurement noise.

### Anomaly Simulator (`anomaly_simulator.py`)
Simulates two types of anomalies:
- **Leaks**: Flow loss on network edges
- **Meter errors**: Measurement errors with different modes (add, multiply, drift)

### Visualizer (`visualizer.py`)
Creates plots for:
- Flow time series with anomaly highlighting
- Flow statistics by node
- Anomaly distribution analysis

### Simulator (`simulator.py`)
Main orchestrator that coordinates all components and manages the simulation workflow.

## Example: Programmatic Usage

```python
from project import FlowSimulator, SimulationConfig

# Create simulator
config = SimulationConfig(num_nodes=10, duration_hours=12)
sim = FlowSimulator(config)

# Run simulation
sim.setup()
sim.run()

# Access data programmatically
for node_id, df in sim.time_series.items():
    print(f"Node {node_id}: {len(df)} samples")
    print(f"Mean flow: {df['flow'].mean():.2f} m³/h")

# Save results
sim.save_data()
sim.visualize()
```

## Performance Notes

- For 100 nodes at 1 Hz sampling for 24 hours: ~8.6M measurements
- Memory usage scales linearly with number of nodes and duration
- Visualization is limited to sample of nodes for clarity
- Use lower sampling frequency (e.g., 0.1 Hz) for longer simulations
