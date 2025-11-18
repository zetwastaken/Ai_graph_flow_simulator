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

### CLI Arguments

View all available options:
```bash
python main.py --help
```

Run with custom parameters:
```bash
python main.py --nodes 50 --topology mesh --duration 48 --anomaly-prob 0.15
```

Example with advanced settings:
```bash
python main.py --nodes 60 --sources 3 --topology grid \
    --duration 48 --sampling 0.2 \
    --anomaly-prob 0.15 --anomaly-rate-multiplier 4 \
    --anomaly-severity 2.5 --export json
```

## Output Files

The simulator generates the following files in the `output/` directory:

- `flow_measurements.csv` - Time series flow data for all nodes
- `anomalies.csv` - List of all generated anomalies
- `topology_info.json` - Network topology information
- `simulation_report.json` - Complete simulation summary
- `flow_plot.png` - Flow visualization for sample nodes
- `flow_statistics.png` - Statistical analysis of flows
- `anomaly_distribution.png` - Anomaly type and magnitude distribution
- `force_directed_graph.png` - **NEW**: Force-directed graph visualization of network topology

## CLI Parameters

### Topology Configuration
- `--nodes N` - Number of nodes in the network (default: 20)
- `--sources N` - Number of source nodes (default: 1)
- `--topology TYPE` - Network topology type: tree, mesh, random, radial, grid (default: tree)

### Time Configuration
- `--duration N` - Simulation duration in hours (default: 24)
- `--sampling F` - Sampling frequency in Hz (default: 0.1)
- `--start-time ISO` - Start time in ISO format (default: today at midnight)

### Flow Configuration
- `--base-flow F` - Base flow rate in m³/h (default: 100.0)
- `--flow-variation F` - Flow variation as a fraction (default: 0.2 = 20%)
- `--noise-std F` - Standard deviation of measurement noise (default: 2.0)

### Anomaly Configuration
- `--anomaly-prob F` - Probability of anomaly occurrence (default: 0.1)
- `--anomaly-rate-multiplier F` - Scale the number of anomalies (default: 1.0)
- `--anomaly-severity F` - Scale anomaly magnitude/severity (default: 1.0)
- `--progressive-leak-prob F` - Probability that a leak is progressive (default: 0.25)
- `--disable-leaks` - Disable leak anomalies
- `--disable-meter-errors` - Disable meter error anomalies

### Output Configuration
- `--output DIR` - Output directory path (default: output)
- `--export FORMAT` - Export format: csv or json (default: csv)
- `--no-visualize` - Skip generating visualization plots

## Programmatic Usage

You can also use the simulator programmatically in your Python code:

```python
from project import FlowSimulator, SimulationConfig
from datetime import datetime

config = SimulationConfig(
    num_nodes=50,                    # Number of consumer nodes
    num_sources=1,                   # Number of supply sources
    topology_type="mesh",            # Topology type
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
- **Force-directed graph**: Network topology visualization showing:
  - Total read data from nodes (sum of all flow measurements in m³)
  - Total flow amounts on edges (average flow rate in m³/h)
  - Color-coded nodes by type (source, hub, consumer)
  - Node size proportional to total flow volume
  - Edge thickness proportional to average flow rate

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

## Force-Directed Graph Visualization

The simulator automatically generates a force-directed graph visualization that provides an intuitive view of the network topology and flow distribution.

### Features

- **Force-directed layout**: Uses NetworkX spring layout algorithm for automatic, physics-based node positioning
- **Node visualization**:
  - **Color**: Red (source), Blue (hubs), Green (consumers)
  - **Size**: Proportional to total flow volume (sum of all readings)
  - **Label**: Shows node ID and total flow in m³
- **Edge visualization**:
  - **Thickness**: Proportional to average flow rate through the connection
  - **Label**: Shows average flow rate in m³/h
  - **Direction**: Arrows indicate flow direction from source to consumers
- **Interactive legend**: Shows node type color coding

### Interpretation

The force-directed graph helps you:
- Identify high-flow vs low-flow nodes by size
- Understand network structure and connectivity
- Visualize flow distribution across the network
- Spot potential bottlenecks (thick edges)
- Analyze hub importance in the distribution network

### Example

The visualization shows:
- The root source node (red) at the center
- Hub nodes (blue) distributing flow to regions
- Consumer nodes (green) at the periphery with varying sizes
- Edge thickness indicating major vs minor flow paths

## Performance Notes

- For 100 nodes at 1 Hz sampling for 24 hours: ~8.6M measurements
- Memory usage scales linearly with number of nodes and duration
- Visualization is limited to sample of nodes for clarity
- Use lower sampling frequency (e.g., 0.1 Hz) for longer simulations
