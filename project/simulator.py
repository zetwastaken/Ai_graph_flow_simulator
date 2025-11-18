"""Main simulator orchestration logic."""

import json
import os
from typing import Dict, Optional

import pandas as pd

from .anomalies import AnomalySimulator
from .config import SimulationConfig
from .data_generator import FlowDataGenerator
from .network_topology import NetworkTopology
from .utils import SimulationDataSaver
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
        self.edge_series = None
        self.anomalies = None
        self.data_saver = SimulationDataSaver(self.config.output_dir, self.config.export_format)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def setup(self):
        """Set up all simulation components."""
        print("Setting up simulation...")
        
        # Create network topology
        self.topology = NetworkTopology(self.config)
        print(f"Network topology created: {self.topology.get_topology_info()}")
        
        # Initialize components
        self.data_generator = FlowDataGenerator(self.config, self.topology)
        self.anomaly_simulator = AnomalySimulator(self.config)
        self.anomaly_simulator.attach_topology(self.topology.graph)
        self.visualizer = FlowVisualizer(self.config.output_dir)
        
        print("Setup completed.")
    
    def run(self):
        """Run the complete simulation."""
        if self.topology is None:
            self.setup()
        
        print("\nGenerating flow data...")
        # Get consumer nodes
        consumer_nodes = self.topology.get_consumers()
        
        # Generate time series for all nodes and edges
        self.time_series, self.edge_series = self.data_generator.generate_time_series(self.topology)
        print(f"Generated {len(self.time_series)} node series and {len(self.edge_series)} edge series")
        
        print("\nGenerating anomalies...")
        # Generate anomalies
        edge_catalog = {
            self.topology.get_edge_id(src, tgt): (src, tgt)
            for src, tgt in self.topology.get_edges()
        }
        self.anomalies = self.anomaly_simulator.generate_anomalies(consumer_nodes, edge_catalog)
        print(f"Generated {len(self.anomalies)} anomalies")
        
        # Apply anomalies to time series
        self.time_series = self.anomaly_simulator.apply_anomalies(self.time_series, self.edge_series)
        print("Anomalies applied to nodes and edges")
        
        print("\nSimulation completed.")
    
    def save_data(self):
        """Save simulation data to files."""
        print("\nSaving data...")
        self.data_saver = SimulationDataSaver(self.config.output_dir, self.config.export_format)
        
        all_data = self.get_node_dataframe()
        edge_data = self.get_edge_dataframe()

        flow_path = self.data_saver.save_node_data(all_data)
        print(f"Flow data saved to {flow_path}")
        edge_path = self.data_saver.save_edge_data(edge_data)
        print(f"Edge flow data saved to {edge_path}")

        anomaly_df = self.anomaly_simulator.get_anomaly_report()
        anomaly_path = self.data_saver.save_anomalies(anomaly_df)
        print(f"Anomaly report saved to {anomaly_path}")

        topology_path = self.data_saver.save_topology(self.topology)
        print(f"Topology info saved to {topology_path}")
    
    def visualize(self):
        """Create visualizations of the simulation data."""
        print("\nCreating visualizations...")
        
        all_data = self.get_node_dataframe()
        edge_data = self.get_edge_dataframe()
        
        # Plot sample of nodes
        sample_nodes = self.topology.get_consumers()[:5]
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
        self.visualizer.plot_force_directed_graph(self.topology.graph, all_data, edge_data)
        print("Force-directed graph visualization created")
        
        print(f"Visualizations saved to {self.config.output_dir}")
    
    def generate_report(self) -> Dict:
        """
        Generate a summary report of the simulation.
        
        Returns:
            Dictionary with simulation statistics
        """
        all_data = self.get_node_dataframe()
        edge_data = self.get_edge_dataframe()
        
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
            'edge_statistics': {
                'mean_flow': float(edge_data['flow'].mean()),
                'std_flow': float(edge_data['flow'].std()),
                'min_flow': float(edge_data['flow'].min()),
                'max_flow': float(edge_data['flow'].max()),
                'total_measurements': len(edge_data)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_node_dataframe(self) -> pd.DataFrame:
        if not self.time_series:
            raise ValueError("No node time series available. Run the simulation first.")
        all_data = pd.concat(self.time_series.values(), ignore_index=True)
        all_data = all_data.sort_values(['timestamp', 'node_id'])
        return all_data

    def get_edge_dataframe(self) -> pd.DataFrame:
        if not self.edge_series:
            raise ValueError("No edge time series available. Run the simulation first.")
        edge_data = pd.concat(self.edge_series.values(), ignore_index=True)
        edge_data = edge_data.sort_values(['timestamp', 'edge_id'])
        return edge_data
    
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
        print("\nEdge Flow Statistics:")
        print(f"  Mean flow: {report['edge_statistics']['mean_flow']:.2f} m³/h")
        print(f"  Std deviation: {report['edge_statistics']['std_flow']:.2f} m³/h")
        print(f"  Min flow: {report['edge_statistics']['min_flow']:.2f} m³/h")
        print(f"  Max flow: {report['edge_statistics']['max_flow']:.2f} m³/h")
        
        print("\nAnomaly Statistics:")
        print(f"  Total anomalies: {report['anomaly_statistics']['total_anomalies']}")
        print(f"  Leaks: {report['anomaly_statistics']['num_leaks']}")
        print(f"  Meter errors: {report['anomaly_statistics']['num_meter_errors']}")
        print(f"  Anomaly percentage: {report['anomaly_statistics']['anomaly_percentage']:.2f}%")
        
        print("\n" + "="*60)
