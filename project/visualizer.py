"""
Visualization module for flow data.
Creates plots and charts for flow analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
import os


class FlowVisualizer:
    """
    Visualizes flow measurement data.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_node_flows(self, time_series: pd.DataFrame, node_ids: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
        """
        Plot flow data for selected nodes.
        
        Args:
            time_series: Combined time series DataFrame
            node_ids: List of node IDs to plot (if None, plot all)
            save_path: Path to save the plot
        """
        if node_ids is not None:
            data = time_series[time_series['node_id'].isin(node_ids)]
        else:
            data = time_series
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for node_id in data['node_id'].unique():
            node_data = data[data['node_id'] == node_id]
            
            # Plot normal flow
            normal_mask = ~node_data['anomaly_active']
            ax.plot(node_data.loc[normal_mask, 'timestamp'], 
                   node_data.loc[normal_mask, 'flow'],
                   label=node_id, alpha=0.7)
            
            # Highlight anomalies
            anomaly_mask = node_data['anomaly_active']
            if anomaly_mask.any():
                ax.scatter(node_data.loc[anomaly_mask, 'timestamp'],
                          node_data.loc[anomaly_mask, 'flow'],
                          color='red', s=5, alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Flow (m³/h)')
        ax.set_title('Flow Measurements Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'flow_plot.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_anomaly_distribution(self, anomaly_df: pd.DataFrame, 
                                  save_path: Optional[str] = None):
        """
        Plot distribution of anomalies.
        
        Args:
            anomaly_df: DataFrame with anomaly information
            save_path: Path to save the plot
        """
        if anomaly_df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Anomaly type distribution
        type_counts = anomaly_df['type'].value_counts()
        axes[0].bar(type_counts.index, type_counts.values, color=['#ff7f0e', '#1f77b4'])
        axes[0].set_xlabel('Anomaly Type')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Anomaly Type Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Anomaly magnitude distribution
        axes[1].hist(anomaly_df['magnitude'], bins=20, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Magnitude')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Anomaly Magnitude Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'anomaly_distribution.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_flow_statistics(self, time_series: pd.DataFrame,
                            save_path: Optional[str] = None):
        """
        Plot flow statistics by node.
        
        Args:
            time_series: Combined time series DataFrame
            save_path: Path to save the plot
        """
        # Calculate statistics per node
        stats = time_series.groupby('node_id')['flow'].agg(['mean', 'std', 'min', 'max'])
        stats = stats.sort_values('mean', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean flow
        axes[0, 0].barh(range(len(stats)), stats['mean'])
        axes[0, 0].set_yticks(range(len(stats)))
        axes[0, 0].set_yticklabels(stats.index, fontsize=8)
        axes[0, 0].set_xlabel('Mean Flow (m³/h)')
        axes[0, 0].set_title('Mean Flow by Node')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Standard deviation
        axes[0, 1].barh(range(len(stats)), stats['std'], color='orange')
        axes[0, 1].set_yticks(range(len(stats)))
        axes[0, 1].set_yticklabels(stats.index, fontsize=8)
        axes[0, 1].set_xlabel('Std Dev (m³/h)')
        axes[0, 1].set_title('Flow Variability by Node')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Min flow
        axes[1, 0].barh(range(len(stats)), stats['min'], color='green')
        axes[1, 0].set_yticks(range(len(stats)))
        axes[1, 0].set_yticklabels(stats.index, fontsize=8)
        axes[1, 0].set_xlabel('Min Flow (m³/h)')
        axes[1, 0].set_title('Minimum Flow by Node')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Max flow
        axes[1, 1].barh(range(len(stats)), stats['max'], color='red')
        axes[1, 1].set_yticks(range(len(stats)))
        axes[1, 1].set_yticklabels(stats.index, fontsize=8)
        axes[1, 1].set_xlabel('Max Flow (m³/h)')
        axes[1, 1].set_title('Maximum Flow by Node')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'flow_statistics.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
