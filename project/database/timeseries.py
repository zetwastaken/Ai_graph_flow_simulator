"""Persistence helpers for time-series data."""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

try:
    from sqlalchemy import create_engine
except Exception:  # pragma: no cover - optional dependency
    create_engine = None  # type: ignore


class TimeseriesClient:
    """Append-only writer that targets TimescaleDB/Influx or local CSV fallback."""

    def __init__(self, url: Optional[str], output_dir: str):
        self.url = url
        self.output_dir = output_dir
        self.engine = create_engine(url) if url and create_engine else None
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.node_file = os.path.join(output_dir, "stream_nodes.csv")
        self.edge_file = os.path.join(output_dir, "stream_edges.csv")

    def _write_dataframe(self, df: pd.DataFrame, table: str, file_path: str):
        if df.empty:
            return
        if self.engine is not None:
            df.to_sql(table, self.engine, if_exists="append", index=False)
        else:
            header = not os.path.exists(file_path)
            df.to_csv(file_path, mode="a", header=header, index=False)

    def write_points(self, payload: Dict):
        nodes_df = pd.DataFrame(payload.get("nodes", []))
        edges_df = pd.DataFrame(payload.get("edges", []))
        self._write_dataframe(nodes_df, "node_measurements", self.node_file)
        self._write_dataframe(edges_df, "edge_measurements", self.edge_file)
