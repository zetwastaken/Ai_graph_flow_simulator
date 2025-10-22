#!/usr/bin/env python3
"""
Example usage of the AI Graph Flow Simulator.
Demonstrates various configurations and use cases.
"""

from datetime import datetime
from project import FlowSimulator, SimulationConfig


def example_basic():
    """Basic simulation with default settings."""
    print("\n" + "="*60)
    print("Example 1: Basic Simulation")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=10,
        duration_hours=12,
        sampling_frequency_hz=0.1,
        output_dir="output/example1"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    sim.save_data()
    sim.visualize()
    sim.print_summary()


def example_high_frequency():
    """Simulation with high sampling frequency."""
    print("\n" + "="*60)
    print("Example 2: High Frequency Sampling")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=5,
        duration_hours=2,
        sampling_frequency_hz=1.0,  # 1 Hz - 1 sample per second
        output_dir="output/example2"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    sim.save_data()
    sim.print_summary()


def example_large_network():
    """Simulation with large network."""
    print("\n" + "="*60)
    print("Example 3: Large Network Simulation")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=50,
        duration_hours=24,
        sampling_frequency_hz=0.05,  # 0.05 Hz = 1 sample every 20 seconds
        base_flow_rate=150.0,
        anomaly_probability=0.15,
        output_dir="output/example3"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    sim.save_data()
    sim.visualize()
    sim.print_summary()


def example_high_anomaly_rate():
    """Simulation with high anomaly occurrence."""
    print("\n" + "="*60)
    print("Example 4: High Anomaly Rate")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=15,
        duration_hours=12,
        sampling_frequency_hz=0.1,
        anomaly_probability=0.3,  # 30% chance per hour
        leak_magnitude_range=(10.0, 25.0),
        meter_error_range=(-10.0, 10.0),
        output_dir="output/example4"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    sim.save_data()
    sim.visualize()
    sim.print_summary()


def example_programmatic_access():
    """Demonstrate programmatic access to simulation data."""
    print("\n" + "="*60)
    print("Example 5: Programmatic Data Access")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=5,
        duration_hours=6,
        sampling_frequency_hz=0.1,
        output_dir="output/example5"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    
    # Access time series data
    print("\nTime Series Data:")
    for node_id, df in list(sim.time_series.items())[:3]:
        print(f"\nNode: {node_id}")
        print(f"  Samples: {len(df)}")
        print(f"  Mean flow: {df['flow'].mean():.2f} m³/h")
        print(f"  Std dev: {df['flow'].std():.2f} m³/h")
        print(f"  Anomalies: {df['anomaly_active'].sum()} samples")
    
    # Access anomaly information
    print("\nAnomaly Information:")
    for anomaly in sim.anomalies[:3]:
        print(f"  {anomaly['id']}: {anomaly['type']} on {anomaly['target_id']}")
        print(f"    Start: {anomaly['start_time']}, Duration: {anomaly['duration_minutes']} min")
        print(f"    Magnitude: {anomaly['magnitude']:.2f}")
    
    # Access topology information
    print("\nTopology Information:")
    info = sim.topology.get_topology_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    sim.save_data()


def example_json_export():
    """Export data in JSON format."""
    print("\n" + "="*60)
    print("Example 6: JSON Export")
    print("="*60)
    
    config = SimulationConfig(
        num_nodes=8,
        duration_hours=6,
        sampling_frequency_hz=0.1,
        export_format="json",  # Export as JSON instead of CSV
        output_dir="output/example6"
    )
    
    sim = FlowSimulator(config)
    sim.setup()
    sim.run()
    sim.save_data()
    sim.print_summary()


def main():
    """Run all examples."""
    import sys
    
    examples = {
        '1': ('Basic Simulation', example_basic),
        '2': ('High Frequency Sampling', example_high_frequency),
        '3': ('Large Network', example_large_network),
        '4': ('High Anomaly Rate', example_high_anomaly_rate),
        '5': ('Programmatic Access', example_programmatic_access),
        '6': ('JSON Export', example_json_export),
        'all': ('All Examples', None)
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        example_num = sys.argv[1]
    else:
        print("\nAI Graph Flow Simulator - Examples")
        print("="*60)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        print("\nUsage: python examples.py [example_number]")
        print("       python examples.py all  (run all examples)")
        return
    
    if example_num == 'all':
        for key, (name, func) in examples.items():
            if func is not None:
                func()
    else:
        _, func = examples[example_num]
        func()


if __name__ == "__main__":
    main()
