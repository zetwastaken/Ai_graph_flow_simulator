"""
Main simulator class that orchestrates the entire simulation.
"""

import os
import json
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

from .config import SimulationConfig
from .network_topology import NetworkTopology
from .data_generator import FlowDataGenerator
from .anomaly_simulator import AnomalySimulator
from .visualizer import FlowVisualizer


class FlowSimulator:
    """
    Main flow simulator that coordinates all components.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the flow simulator.
        
        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        self.topology = None
        self.data_generator = None
        self.anomaly_simulator = None
        self.visualizer = None
        self.time_series = None
        self.anomalies = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def setup(self):
        """Set up all simulation components."""
        print("Setting up simulation...")
        
        # Create network topology
        self.topology = NetworkTopology(
            num_nodes=self.config.num_nodes,
            num_sources=self.config.num_sources
        )
        print(f"Network topology created: {self.topology.get_topology_info()}")
        
        # Initialize components
        self.data_generator = FlowDataGenerator(self.config)
        self.anomaly_simulator = AnomalySimulator(self.config)
        self.visualizer = FlowVisualizer(self.config.output_dir)
        
        print("Setup completed.")
    
    def run(self):
        """Run the complete simulation."""
        if self.topology is None:
            self.setup()
        
        print("\nGenerating flow data...")
        # Get consumer nodes
        consumer_nodes = self.topology.get_consumers()
        
        # Generate time series for all nodes
        self.time_series = self.data_generator.generate_time_series(consumer_nodes)
        print(f"Generated {len(self.time_series)} time series")
        
        print("\nGenerating anomalies...")
        # Generate anomalies
        edge_ids = [self.topology.get_edge_id(src, tgt) for src, tgt in self.topology.get_edges()]
        self.anomalies = self.anomaly_simulator.generate_anomalies(consumer_nodes, edge_ids)
        print(f"Generated {len(self.anomalies)} anomalies")
        
        # Apply anomalies to time series
        self.time_series = self.anomaly_simulator.apply_anomalies(self.time_series)
        print("Anomalies applied to data")
        
        print("\nSimulation completed.")
    
    def save_data(self):
        """Save simulation data to files."""
        print("\nSaving data...")
        
        # Combine all time series
        all_data = pd.concat(self.time_series.values(), ignore_index=True)
        all_data = all_data.sort_values(['timestamp', 'node_id'])
        
        # Save flow measurements
        if self.config.export_format == 'csv':
            flow_path = os.path.join(self.config.output_dir, 'flow_measurements.csv')
            all_data.to_csv(flow_path, index=False)
            print(f"Flow data saved to {flow_path}")
        elif self.config.export_format == 'json':
            flow_path = os.path.join(self.config.output_dir, 'flow_measurements.json')
            all_data.to_json(flow_path, orient='records', date_format='iso')
            print(f"Flow data saved to {flow_path}")
        
        # Save anomaly report (always save, even if empty)
        anomaly_df = self.anomaly_simulator.get_anomaly_report()
        anomaly_path = os.path.join(self.config.output_dir, 'anomalies.csv')
        anomaly_df.to_csv(anomaly_path, index=False)
        print(f"Anomaly report saved to {anomaly_path}")
        
        # Save topology information
        topology_info = self.topology.get_topology_info()
        topology_path = os.path.join(self.config.output_dir, 'topology_info.json')
        with open(topology_path, 'w') as f:
            json.dump(topology_info, f, indent=2)
        print(f"Topology info saved to {topology_path}")
    
    def visualize(self):
        """Create visualizations of the simulation data."""
        print("\nCreating visualizations...")
        
        # Combine all time series for visualization
        all_data = pd.concat(self.time_series.values(), ignore_index=True)
        
        # Plot sample of nodes
        sample_nodes = list(self.time_series.keys())[:5]
        self.visualizer.plot_node_flows(all_data, sample_nodes)
        print("Flow plot created")
        
        # Plot flow statistics
        self.visualizer.plot_flow_statistics(all_data)
        print("Statistics plot created")
        
        # Plot anomaly distribution (if there are any anomalies)
        if self.anomalies:
            anomaly_df = self.anomaly_simulator.get_anomaly_report()
            self.visualizer.plot_anomaly_distribution(anomaly_df)
            print("Anomaly distribution plot created")
        else:
            print("No anomalies to plot")
        
        # Create force-directed graph visualization
        self.visualizer.plot_force_directed_graph(self.topology.graph, all_data)
        print("Force-directed graph visualization created")
        
        print(f"Visualizations saved to {self.config.output_dir}")
    
    def generate_report(self) -> Dict:
        """
        Generate a summary report of the simulation.
        
        Returns:
            Dictionary with simulation statistics
        """
        all_data = pd.concat(self.time_series.values(), ignore_index=True)
        
        report = {
            'simulation_info': {
                'start_time': self.config.start_time.isoformat(),
                'end_time': self.config.end_time.isoformat(),
                'duration_hours': self.config.duration_hours,
                'sampling_frequency_hz': self.config.sampling_frequency_hz,
                'total_samples': self.config.total_samples
            },
            'topology_info': self.topology.get_topology_info(),
            'flow_statistics': {
                'mean_flow': float(all_data['flow'].mean()),
                'std_flow': float(all_data['flow'].std()),
                'min_flow': float(all_data['flow'].min()),
                'max_flow': float(all_data['flow'].max()),
                'total_measurements': len(all_data)
            },
            'anomaly_statistics': {
                'total_anomalies': len(self.anomalies),
                'num_leaks': sum(1 for a in self.anomalies if a['type'] == 'leak'),
                'num_meter_errors': sum(1 for a in self.anomalies if a['type'] == 'meter_error'),
                'anomaly_percentage': float(all_data['anomaly_active'].mean() * 100)
            }
        }
        
        # Save report
        report_path = os.path.join(self.config.output_dir, 'simulation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSimulation report saved to {report_path}")
        
        return report
    
    def print_summary(self):
        """Print a summary of the simulation."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        print("\nSimulation Parameters:")
        print(f"  Duration: {report['simulation_info']['duration_hours']} hours")
        print(f"  Sampling frequency: {report['simulation_info']['sampling_frequency_hz']} Hz")
        print(f"  Total samples: {report['simulation_info']['total_samples']}")
        
        print("\nNetwork Topology:")
        print(f"  Total nodes: {report['topology_info']['num_nodes']}")
        print(f"  Consumer nodes: {report['topology_info']['num_consumers']}")
        print(f"  Total edges: {report['topology_info']['num_edges']}")
        
        print("\nFlow Statistics:")
        print(f"  Mean flow: {report['flow_statistics']['mean_flow']:.2f} m³/h")
        print(f"  Std deviation: {report['flow_statistics']['std_flow']:.2f} m³/h")
        print(f"  Min flow: {report['flow_statistics']['min_flow']:.2f} m³/h")
        print(f"  Max flow: {report['flow_statistics']['max_flow']:.2f} m³/h")
        
        print("\nAnomaly Statistics:")
        print(f"  Total anomalies: {report['anomaly_statistics']['total_anomalies']}")
        print(f"  Leaks: {report['anomaly_statistics']['num_leaks']}")
        print(f"  Meter errors: {report['anomaly_statistics']['num_meter_errors']}")
        print(f"  Anomaly percentage: {report['anomaly_statistics']['anomaly_percentage']:.2f}%")
        
        print("\n" + "="*60)
