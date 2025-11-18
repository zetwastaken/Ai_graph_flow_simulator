"""Persistence helpers for simulator artifacts."""

from __future__ import annotations

import json
import os
import pandas as pd

from ..network_topology import NetworkTopology


class SimulationDataSaver:
    """Encapsulates saving flow series, anomalies, and topology metadata."""

    def __init__(self, output_dir: str, export_format: str = "csv"):
        self.output_dir = output_dir
        self.export_format = export_format
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Flow persistence helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, basename: str) -> str:
        extension = "json" if self.export_format == "json" else "csv"
        return os.path.join(self.output_dir, f"{basename}.{extension}")

    def save_node_data(self, df: pd.DataFrame) -> str:
        path = self._resolve_path("flow_measurements")
        if self.export_format == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            df.to_csv(path, index=False)
        return path

    def save_edge_data(self, df: pd.DataFrame) -> str:
        path = os.path.join(self.output_dir, f"edge_flows.{self.export_format}")
        if self.export_format == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def save_anomalies(self, df: pd.DataFrame) -> str:
        path = os.path.join(self.output_dir, "anomalies.csv")
        df.to_csv(path, index=False)
        return path

    def save_topology(self, topology: NetworkTopology) -> str:
        topology_info = topology.get_topology_info()
        payload = {
            "metadata": topology_info,
            "nodes": [
                {
                    "id": node,
                    "type": topology.get_node_type(node),
                }
                for node in topology.get_nodes()
            ],
            "edges": [
                {
                    "id": topology.get_edge_id(src, tgt),
                    "source": src,
                    "target": tgt,
                    "length": topology.graph.edges[src, tgt].get("length"),
                }
                for src, tgt in topology.get_edges()
            ],
        }
        path = os.path.join(self.output_dir, "topology_info.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path
