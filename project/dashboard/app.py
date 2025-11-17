"""Dash dashboard for interactive visualization."""

from __future__ import annotations

import json
import os
from typing import List

import dash
import dash_cytoscape as cyto
import plotly.express as px
import requests
from dash import Dash, Input, Output, dcc, html
import pandas as pd

SNAPSHOT_URL = os.getenv("SIMULATOR_SNAPSHOT_URL", "http://localhost:8080/snapshot")
DATA_DIR = os.getenv("SIMULATOR_OUTPUT", "output")


class DashboardDataSource:
    def __init__(self, data_dir: str, snapshot_url: str):
        self.data_dir = data_dir
        self.snapshot_url = snapshot_url

    def _load_static(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "flow_measurements.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def fetch(self) -> pd.DataFrame:
        try:
            response = requests.get(self.snapshot_url, timeout=2)
            response.raise_for_status()
            payload = response.json()
            rows = []
            for frame in payload:
                rows.extend(frame.get("nodes", []))
            return pd.DataFrame(rows)
        except Exception:
            return self._load_static()

    def load_topology(self) -> List[dict]:
        topo_path = os.path.join(self.data_dir, "topology_info.json")
        if not os.path.exists(topo_path):
            return []
        with open(topo_path, "r") as f:
            info = json.load(f)
        nodes = [
            {"data": {"id": item["id"], "label": f"{item['id']} ({item['type']})"}}
            for item in info.get("nodes", [])
        ]
        edges = [
            {
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": f"{edge['length']} m"
                }
            }
            for edge in info.get("edges", [])
        ]
        return nodes + edges


def create_app() -> Dash:
    dash_app = Dash(__name__)
    source = DashboardDataSource(DATA_DIR, SNAPSHOT_URL)
    topo_elements = source.load_topology()

    dash_app.layout = html.Div(
        [
            html.H2("AI Flow Simulator Dashboard"),
            html.Div(
                [
                    dcc.Dropdown(id="node-filter", placeholder="Select node"),
                    dcc.Interval(id="refresh", interval=5000, n_intervals=0),
                ]
            ),
            dcc.Graph(id="flow-chart"),
            cyto.Cytoscape(
                id="topology", elements=topo_elements, style={"height": "400px"},
                layout={"name": "breadthfirst"}
            ),
        ]
    )

    @dash_app.callback(
        Output("flow-chart", "figure"),
        Output("node-filter", "options"),
        Input("refresh", "n_intervals"),
        Input("node-filter", "value"),
        prevent_initial_call=False,
    )
    def update_chart(_: int, selected_node: str):  # type: ignore[override]
        df = source.fetch()
        if df.empty:
            return px.line(title="Awaiting data"), []
        options = sorted(df["node_id"].unique())
        filtered = df if not selected_node else df[df["node_id"] == selected_node]
        fig = px.line(filtered, x="timestamp", y="flow", color="node_id", title="Flow over time")
        return fig, options

    return dash_app


if __name__ == "__main__":
    create_app().run_server(debug=True)
