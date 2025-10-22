# -*- coding: utf-8 -*-
"""
Visualizes the simulation data.
"""
import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    """
    Creates plots for the simulation data.
    """
    def __init__(self, data_filepath):
        """
        Initializes the Plotter.

        Args:
            data_filepath (str): Path to the CSV file with simulation data.
        """
        self.data_filepath = data_filepath
        self.data = None

    def load_data(self):
        """
        Loads data from the CSV file.
        """
        self.data = pd.DataFrame()
        try:
            self.data = pd.read_csv(self.data_filepath)
            print("Data loaded for plotting.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_filepath}")

    def plot_flow_for_node(self, node_id, save_path=None):
        """
        Plots the flow for a specific node over time.

        Args:
            node_id (int): The ID of the node to plot.
            save_path (str, optional): Path to save the plot image. Defaults to None (shows plot).
        """
        if self.data.empty:
            print("Data is not loaded. Cannot plot.")
            return

        node_data = self.data[self.data['node'] == node_id]
        if node_data.empty:
            print(f"No data for node {node_id}")
            return

        plt.figure(figsize=(15, 6))
        plt.plot(node_data['timestamp'], node_data['flow'], label=f'Flow for Node {node_id}')
        
        anomalies = node_data[node_data['anomaly'] != 'none']
        if not anomalies.empty:
            plt.scatter(anomalies['timestamp'], anomalies['flow'], color='red', zorder=5, label='Anomaly')

        plt.xlabel("Time (seconds)")
        plt.ylabel("Flow")
        plt.title(f"Flow Over Time for Node {node_id}")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_all_flows(self, save_path=None):
        """
        Plots all flows on a single graph.
        """
        if self.data.empty:
            print("Data is not loaded. Cannot plot.")
            return
            
        plt.figure(figsize=(15, 8))
        for node_id in self.data['node'].unique():
            node_data = self.data[self.data['node'] == node_id]
            plt.plot(node_data['timestamp'], node_data['flow'], label=f'Node {node_id}')

        plt.xlabel("Time (seconds)")
        plt.ylabel("Flow")
        plt.title("All Node Flows Over Time")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
