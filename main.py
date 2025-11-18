#!/usr/bin/env python3
"""
Enhanced CLI entry point for AI Graph Flow Simulator.
Pure simulation without web UI components.
"""

import argparse
from datetime import datetime
from project import FlowSimulator, SimulationConfig


def main():
    """Run the flow simulator with CLI argument support."""
    
    parser = argparse.ArgumentParser(
        description="AI Graph Flow Simulator - CLI Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --nodes 50 --topology mesh --duration 48
  %(prog)s --nodes 30 --sources 2 --anomaly-prob 0.2 --export json
  %(prog)s --topology radial --duration 72 --sampling 0.05 --no-visualize
        """
    )
    
    # Topology arguments
    topo_group = parser.add_argument_group('Topology Configuration')
    topo_group.add_argument(
        "--nodes", type=int, default=20,
        help="Number of nodes in the network (default: 20)"
    )
    topo_group.add_argument(
        "--sources", type=int, default=1,
        help="Number of source nodes (default: 1)"
    )
    topo_group.add_argument(
        "--topology", 
        choices=["tree", "mesh", "random", "radial", "grid"], 
        default="tree",
        help="Network topology type (default: tree)"
    )
    
    # Time arguments
    time_group = parser.add_argument_group('Time Configuration')
    time_group.add_argument(
        "--duration", type=int, default=24,
        help="Simulation duration in hours (default: 24)"
    )
    time_group.add_argument(
        "--sampling", type=float, default=0.1,
        help="Sampling frequency in Hz (default: 0.1, i.e., 1 sample per 10 seconds)"
    )
    time_group.add_argument(
        "--start-time", type=str, default=None,
        help="Start time in ISO format (default: today at midnight)"
    )
    
    # Flow arguments
    flow_group = parser.add_argument_group('Flow Configuration')
    flow_group.add_argument(
        "--base-flow", type=float, default=100.0,
        help="Base flow rate in m³/h (default: 100.0)"
    )
    flow_group.add_argument(
        "--flow-variation", type=float, default=0.2,
        help="Flow variation as a fraction (default: 0.2 = 20%%)"
    )
    flow_group.add_argument(
        "--noise-std", type=float, default=2.0,
        help="Standard deviation of measurement noise (default: 2.0)"
    )
    
    # Anomaly arguments
    anomaly_group = parser.add_argument_group('Anomaly Configuration')
    anomaly_group.add_argument(
        "--anomaly-prob", type=float, default=0.1,
        help="Probability of anomaly occurrence (default: 0.1)"
    )
    anomaly_group.add_argument(
        "--anomaly-rate-multiplier", type=float, default=1.0,
        help="Scale the number of anomalies (default: 1.0)"
    )
    anomaly_group.add_argument(
        "--anomaly-severity", type=float, default=1.0,
        help="Scale anomaly magnitude/severity (default: 1.0)"
    )
    anomaly_group.add_argument(
        "--disable-leaks", action="store_true",
        help="Disable leak anomalies"
    )
    anomaly_group.add_argument(
        "--disable-meter-errors", action="store_true",
        help="Disable meter error anomalies"
    )
    anomaly_group.add_argument(
        "--progressive-leak-prob", type=float, default=0.25,
        help="Probability that a leak is progressive rather than constant (default: 0.25)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output", default="output",
        help="Output directory path (default: output)"
    )
    output_group.add_argument(
        "--export", 
        choices=["csv", "json"], 
        default="csv",
        help="Export format for data files (default: csv)"
    )
    output_group.add_argument(
        "--no-visualize", action="store_true",
        help="Skip generating visualization plots"
    )
    
    args = parser.parse_args()
    
    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.fromisoformat(args.start_time)
        except ValueError:
            print(f"Error: Invalid start time format: {args.start_time}")
            print("Use ISO format like: 2025-01-01T00:00:00")
            return 1
    else:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Create configuration
    config = SimulationConfig(
        num_nodes=args.nodes,
        num_sources=args.sources,
        topology_type=args.topology,
        start_time=start_time,
        duration_hours=args.duration,
        sampling_frequency_hz=args.sampling,
        base_flow_rate=args.base_flow,
        flow_variation=args.flow_variation,
        noise_std=args.noise_std,
        anomaly_probability=args.anomaly_prob,
        anomaly_rate_multiplier=args.anomaly_rate_multiplier,
        anomaly_severity=args.anomaly_severity,
        progressive_leak_probability=args.progressive_leak_prob,
        enable_leaks=not args.disable_leaks,
        enable_meter_errors=not args.disable_meter_errors,
        output_dir=args.output,
        export_format=args.export,
    )
    
    print("AI Graph Flow Simulator - CLI Mode")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Topology: {args.topology} with {args.nodes} nodes, {args.sources} sources")
    print(f"  Duration: {args.duration} hours @ {args.sampling} Hz sampling")
    print(f"  Anomalies: {args.anomaly_prob:.1%} probability"
          f"{'' if not args.disable_leaks else ' (leaks disabled)'}"
          f"{'' if not args.disable_meter_errors else ' (meter errors disabled)'}")
    print(f"  Output: {args.output}/ ({args.export} format)")
    print("=" * 60)
    print()
    
    # Create and run simulator
    simulator = FlowSimulator(config)
    
    # Setup the simulation
    print("Setting up simulation...")
    simulator.setup()
    
    # Run the simulation
    print("Running simulation...")
    simulator.run()
    
    # Save results
    print("Saving data...")
    simulator.save_data()
    
    # Create visualizations
    if not args.no_visualize:
        print("Generating visualizations...")
        simulator.visualize()
    
    # Print summary
    print()
    simulator.print_summary()
    
    print(f"\n✓ Simulation complete! Results saved to '{args.output}/' directory.")
    return 0


if __name__ == "__main__":
    exit(main())
