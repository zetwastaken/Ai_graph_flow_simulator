"""Visualization utilities for flow data."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


class FlowVisualizer:
    """Visualizes flow measurement data and metadata."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_node_flows(
        self,
        time_series: pd.DataFrame,
        node_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        if node_ids is not None:
            data = time_series[time_series["node_id"].isin(node_ids)]
        else:
            data = time_series

        fig, ax = plt.subplots(figsize=(14, 6))

        for node_id in data["node_id"].unique():
            node_data = data[data["node_id"] == node_id]
            normal_mask = ~node_data["anomaly_active"]
            ax.plot(
                node_data.loc[normal_mask, "timestamp"],
                node_data.loc[normal_mask, "flow"],
                label=node_id,
                alpha=0.7,
            )

            anomaly_mask = node_data["anomaly_active"]
            if anomaly_mask.any():
                ax.scatter(
                    node_data.loc[anomaly_mask, "timestamp"],
                    node_data.loc[anomaly_mask, "flow"],
                    color="red",
                    s=5,
                    alpha=0.5,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Flow (m³/h)")
        ax.set_title("Flow Measurements Over Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output = save_path or os.path.join(self.output_dir, "flow_plot.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_anomaly_distribution(self, anomaly_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        if anomaly_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        type_counts = anomaly_df["type"].value_counts()
        axes[0].bar(type_counts.index, type_counts.values, color=["#ff7f0e", "#1f77b4"])
        axes[0].set_xlabel("Anomaly Type")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Anomaly Type Distribution")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].hist(anomaly_df["magnitude"], bins=20, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Magnitude")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Anomaly Magnitude Distribution")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output = save_path or os.path.join(self.output_dir, "anomaly_distribution.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_flow_statistics(self, time_series: pd.DataFrame, save_path: Optional[str] = None) -> None:
        stats = time_series.groupby("node_id")["flow"].agg(["mean", "std", "min", "max"])
        stats = stats.sort_values("mean", ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].barh(range(len(stats)), stats["mean"])
        axes[0, 0].set_yticks(range(len(stats)))
        axes[0, 0].set_yticklabels(stats.index, fontsize=8)
        axes[0, 0].set_xlabel("Mean Flow (m³/h)")
        axes[0, 0].set_title("Mean Flow by Node")
        axes[0, 0].grid(True, alpha=0.3, axis="x")

        axes[0, 1].barh(range(len(stats)), stats["std"], color="orange")
        axes[0, 1].set_yticks(range(len(stats)))
        axes[0, 1].set_yticklabels(stats.index, fontsize=8)
        axes[0, 1].set_xlabel("Std Dev (m³/h)")
        axes[0, 1].set_title("Flow Variability by Node")
        axes[0, 1].grid(True, alpha=0.3, axis="x")

        axes[1, 0].barh(range(len(stats)), stats["min"], color="green")
        axes[1, 0].set_yticks(range(len(stats)))
        axes[1, 0].set_yticklabels(stats.index, fontsize=8)
        axes[1, 0].set_xlabel("Min Flow (m³/h)")
        axes[1, 0].set_title("Minimum Flow by Node")
        axes[1, 0].grid(True, alpha=0.3, axis="x")

        axes[1, 1].barh(range(len(stats)), stats["max"], color="red")
        axes[1, 1].set_yticks(range(len(stats)))
        axes[1, 1].set_yticklabels(stats.index, fontsize=8)
        axes[1, 1].set_xlabel("Max Flow (m³/h)")
        axes[1, 1].set_title("Maximum Flow by Node")
        axes[1, 1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        output = save_path or os.path.join(self.output_dir, "flow_statistics.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_force_directed_graph(
        self,
        topology_graph: nx.DiGraph,
        node_series: pd.DataFrame,
        edge_series: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        node_totals = node_series.groupby("node_id")["flow"].sum().to_dict()
        node_avg_flow = node_series.groupby("node_id")["flow"].mean().to_dict()
        graph = topology_graph.copy()

        for node in graph.nodes():
            graph.nodes[node]["total_flow"] = node_totals.get(node, 0.0)
            graph.nodes[node]["avg_flow"] = node_avg_flow.get(node, 0.0)

        edge_total_flows: Dict[tuple, float] = {}
        edge_avg_flows: Dict[tuple, float] = {}
        meta = edge_series[["edge_id", "source", "target"]].drop_duplicates().set_index("edge_id")
        totals = edge_series.groupby("edge_id")["flow"].sum().to_dict()
        avgs = edge_series.groupby("edge_id")["flow"].mean().to_dict()
        for edge_id, row in meta.iterrows():
            edge = (row["source"], row["target"])
            edge_total_flows[edge] = totals.get(edge_id, 0.0)
            edge_avg_flows[edge] = avgs.get(edge_id, 0.0)

        fig, ax = plt.subplots(figsize=(20, 16))
        try:
            pos = nx.kamada_kawai_layout(graph, scale=5.0)
        except Exception:
            pos = nx.spring_layout(graph, k=5.0, iterations=150, seed=42)

        def _apply_repulsion(
            positions: Dict[str, List[float]], min_distance: float = 1.5, iterations: int = 200
        ) -> Dict[str, List[float]]:
            rng = np.random.default_rng(42)
            adjusted = {node: np.array(coord, dtype=float) for node, coord in positions.items()}
            nodes_list = list(adjusted.keys())
            for _ in range(iterations):
                moved = False
                for i, node_u in enumerate(nodes_list):
                    for node_v in nodes_list[i + 1 :]:
                        delta = adjusted[node_u] - adjusted[node_v]
                        distance = np.linalg.norm(delta)
                        if distance < 1e-6:
                            delta = rng.normal(size=2)
                            distance = np.linalg.norm(delta)
                        if distance < min_distance and distance > 0:
                            move_vec = (min_distance - distance) * (delta / distance) * 0.5
                            adjusted[node_u] += move_vec
                            adjusted[node_v] -= move_vec
                            moved = True
                if not moved:
                    break
            return {node: coord.tolist() for node, coord in adjusted.items()}

        pos = _apply_repulsion(pos)

        node_colors: List[str] = []
        node_sizes: List[float] = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get("node_type", "unknown")
            total_flow = graph.nodes[node].get("total_flow", 0)
            if node_type == "source":
                node_colors.append("#ff4444")
                node_sizes.append(3000)
            elif node_type == "hub":
                node_colors.append("#4444ff")
                node_sizes.append(2000)
            elif node_type == "consumer":
                node_colors.append("#44ff44")
                if node_totals:
                    max_flow = max(node_totals.values()) or 1
                    size = 500 + (total_flow / max_flow * 1500)
                else:
                    size = 500
                node_sizes.append(size)
            else:
                node_colors.append("#888888")
                node_sizes.append(500)

        edge_widths: List[float] = []
        edge_colors: List[str] = []
        max_edge_flow = max(edge_avg_flows.values()) if edge_avg_flows else 1
        for edge in graph.edges():
            flow = edge_avg_flows.get(edge, 0.0)
            width = 1 + (flow / max_edge_flow * 7) if flow > 0 else 1
            edge_widths.append(width)
            edge_colors.append("#666666")

        nx.draw_networkx_edges(
            graph,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            ax=ax,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)

        node_labels: Dict[str, str] = {}
        for node in graph.nodes():
            total_flow = graph.nodes[node].get("total_flow", 0)
            node_labels[node] = f"{node}\n{total_flow:.0f} m³" if total_flow > 0 else node
        nx.draw_networkx_labels(graph, pos, node_labels, font_size=9, font_weight="bold", ax=ax)

        edge_labels: Dict[tuple, str] = {}
        for edge in graph.edges():
            total_flow = edge_total_flows.get(edge, 0)
            avg_flow = edge_avg_flows.get(edge, 0)
            if total_flow > 0 or avg_flow > 0:
                edge_labels[edge] = f"{total_flow:.0f} m³\n({avg_flow:.1f} m³/h)"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8, font_color="#333333", ax=ax)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#ff4444", label="Source Node"),
            Patch(facecolor="#4444ff", label="Hub Node"),
            Patch(facecolor="#44ff44", label="Consumer Node"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        ax.set_title(
            "Force-Directed Network Topology (Kamada-Kawai Layout)\n"
            "Node size = Total flow volume | Edge thickness = Average flow rate\n"
            "Edge labels show: Total volume (Average rate)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")
        plt.tight_layout()
        output = save_path or os.path.join(self.output_dir, "force_directed_graph.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()
