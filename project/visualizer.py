"""
Visualization module for flow data.
Creates plots and charts for flow analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Optional, Dict
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
    
    def plot_force_directed_graph(self, topology_graph: nx.DiGraph, 
                                  time_series: pd.DataFrame,
                                  save_path: Optional[str] = None):
        """
        Create a force-directed graph visualization showing network topology
        with total read data from nodes and total amounts on edges.
        
        Args:
            topology_graph: NetworkX graph representing the network topology
            time_series: Combined time series DataFrame with flow measurements
            save_path: Path to save the plot
        """
        # Calculate total flow per node (sum of all readings)
        node_totals = time_series.groupby('node_id')['flow'].sum().to_dict()
        
        # Calculate average flow per node for edge flow estimation
        node_avg_flow = time_series.groupby('node_id')['flow'].mean().to_dict()
        
        # Create a copy of the graph to add attributes
        G = topology_graph.copy()
        
        # Add total flow as node attribute
        for node in G.nodes():
            if node in node_totals:
                G.nodes[node]['total_flow'] = node_totals[node]
                G.nodes[node]['avg_flow'] = node_avg_flow[node]
            else:
                # For source/hub nodes without measurements, estimate from downstream
                G.nodes[node]['total_flow'] = 0
                G.nodes[node]['avg_flow'] = 0
        
        # Calculate edge flows (sum of downstream node flows)
        # For total amounts, we'll sum all flow that passed through each edge
        edge_total_flows = {}
        edge_avg_flows = {}
        
        for edge in G.edges():
            source, target = edge
            # Edge flow is approximately the flow through the target node
            if target in node_avg_flow:
                edge_avg_flows[edge] = node_avg_flow[target]
                edge_total_flows[edge] = node_totals.get(target, 0)
            else:
                # For edges to hubs, sum all downstream consumer flows
                downstream_consumers = list(nx.descendants(G, target))
                edge_avg_flow = sum(node_avg_flow.get(n, 0) for n in downstream_consumers)
                edge_total_flow = sum(node_totals.get(n, 0) for n in downstream_consumers)
                edge_avg_flows[edge] = edge_avg_flow
                edge_total_flows[edge] = edge_total_flow
        
        # Create figure with larger size for better spacing
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Use Kamada-Kawai layout for better edge crossing reduction
        # This algorithm minimizes energy and reduces edge crossings
        # Increased scale parameter spreads nodes further apart
        try:
            pos = nx.kamada_kawai_layout(G, scale=5.0)
        except:
            # Fallback to spring layout if Kamada-Kawai fails
            # Higher k value pushes nodes further apart
            pos = nx.spring_layout(G, k=5.0, iterations=150, seed=42)
        
        # Apply an extra repulsion step so nodes stay separated even in dense areas
        def _apply_repulsion(positions: Dict[str, List[float]],
                             min_distance: float = 1.5,
                             iterations: int = 200) -> Dict[str, List[float]]:
            """Push nodes apart when they are closer than the requested distance."""
            rng = np.random.default_rng(42)
            adjusted = {node: np.array(coord, dtype=float) for node, coord in positions.items()}
            nodes_list = list(adjusted.keys())
            for _ in range(iterations):
                moved = False
                for i, node_u in enumerate(nodes_list):
                    for node_v in nodes_list[i + 1:]:
                        delta = adjusted[node_u] - adjusted[node_v]
                        distance = np.linalg.norm(delta)
                        if distance < 1e-6:
                            # Random tiny nudge breaks perfect overlap
                            delta = rng.normal(size=2)
                            distance = np.linalg.norm(delta)
                        if distance < min_distance and distance > 0:
                            # Push both nodes away from each other
                            move_vec = (min_distance - distance) * (delta / distance) * 0.5
                            adjusted[node_u] += move_vec
                            adjusted[node_v] -= move_vec
                            moved = True
                if not moved:
                    break
            return {node: coord.tolist() for node, coord in adjusted.items()}

        pos = _apply_repulsion(pos)

        # Prepare node colors and sizes based on flow
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'unknown')
            total_flow = G.nodes[node].get('total_flow', 0)
            
            if node_type == 'source':
                node_colors.append('#ff4444')  # Red for source
                node_sizes.append(3000)
            elif node_type == 'hub':
                node_colors.append('#4444ff')  # Blue for hubs
                node_sizes.append(2000)
            elif node_type == 'consumer':
                node_colors.append('#44ff44')  # Green for consumers
                # Size based on total flow (normalized)
                if total_flow > 0:
                    size = 500 + (total_flow / max(node_totals.values()) * 1500)
                else:
                    size = 500
                node_sizes.append(size)
            else:
                node_colors.append('#888888')
                node_sizes.append(500)
        
        # Draw edges with varying thickness based on flow
        edge_widths = []
        edge_colors = []
        for edge in G.edges():
            flow = edge_avg_flows.get(edge, 0)
            if flow > 0:
                # Normalize width between 1 and 8
                max_flow = max(edge_avg_flows.values()) if edge_avg_flows else 1
                width = 1 + (flow / max_flow * 7)
            else:
                width = 1
            edge_widths.append(width)
            edge_colors.append('#666666')
        
        # Draw the graph
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,  # type: ignore[arg-type]
                              alpha=0.6, arrows=True, arrowsize=20, ax=ax,
                              arrowstyle='->', connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              alpha=0.9, ax=ax)
        
        # Draw node labels with total flow
        node_labels = {}
        for node in G.nodes():
            total_flow = G.nodes[node].get('total_flow', 0)
            if total_flow > 0:
                node_labels[node] = f"{node}\n{total_flow:.0f} m³"
            else:
                node_labels[node] = node
        
        nx.draw_networkx_labels(G, pos, node_labels, font_size=9,
                               font_weight='bold', ax=ax)
        
        # Draw edge labels with total flow amounts and average rate
        edge_labels = {}
        for edge in G.edges():
            total_flow = edge_total_flows.get(edge, 0)
            avg_flow = edge_avg_flows.get(edge, 0)
            if total_flow > 0 or avg_flow > 0:
                edge_labels[edge] = f"{total_flow:.0f} m³\n({avg_flow:.1f} m³/h)"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8,
                                    font_color='#333333', ax=ax)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff4444', label='Source Node'),
            Patch(facecolor='#4444ff', label='Hub Node'),
            Patch(facecolor='#44ff44', label='Consumer Node')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add title and description
        ax.set_title('Force-Directed Network Topology (Kamada-Kawai Layout)\n' + 
                    'Node size = Total flow volume | Edge thickness = Average flow rate\n' +
                    'Edge labels show: Total volume (Average rate)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'force_directed_graph.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
