#!/usr/bin/env python3
"""
Main entry point for the AI Graph Flow Simulator.
Demonstrates basic usage of the simulator.
"""

from datetime import datetime
from project import FlowSimulator, SimulationConfig


def main():
    """Run the flow simulator with default configuration."""
    
    print("AI Graph Flow Simulator")
    print("=" * 60)
    
    # Create configuration
    config = SimulationConfig(
        num_nodes=20,
        num_sources=1,
        start_time=datetime(2025, 1, 1, 0, 0, 0),
        duration_hours=24,
        sampling_frequency_hz=0.1,  # Sample every 10 seconds for demonstration
        base_flow_rate=100.0,
        flow_variation=0.2,
        noise_std=2.0,
        anomaly_probability=0.05,
        output_dir="output",
        export_format="csv"
    )
    
    # Create and run simulator
    simulator = FlowSimulator(config)
    
    # Setup the simulation
    simulator.setup()
    
    # Run the simulation
    simulator.run()
    
    # Save results
    simulator.save_data()
    
    # Create visualizations
    simulator.visualize()
    
    # Print summary
    simulator.print_summary()
    
    print("\nSimulation complete! Check the 'output' directory for results.")


if __name__ == "__main__":
    main()
