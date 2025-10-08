"""Interactive Plotly viewer for simulated network flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from sim.generator import Graph, force_directed_layout


@dataclass
class _NodeInfo:
    """Lightweight descriptor used to attach metadata to node ids."""

    node_id: str
    kind: str
    meter_id: str | None


def _compute_positions(graph: Graph) -> Dict[str, Tuple[float, float]]:
    """Compute deterministic layout positions for the interactive figure."""

    node_count = len(graph.nodes)
    if node_count == 0:
        return {}
    base_area = max(120000.0, 380.0 * node_count * node_count)
    positions = force_directed_layout(
        graph,
        area=base_area,
        seed=0,
        spread=6.5,
        min_distance=2.5,
        edge_separation=5.0,
    )
    if not positions:
        return {}
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    scale_x = max(max(xs) - min(xs), 1.0)
    scale_y = max(max(ys) - min(ys), 1.0)
    scale = max(scale_x, scale_y)
    return {
        node_id: ((x - cx) / scale * 10.0, (y - cy) / scale * 10.0)
        for node_id, (x, y) in positions.items()
    }


def _normalise(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _collect_node_info(graph: Graph) -> Dict[str, _NodeInfo]:
    return {
        node_id: _NodeInfo(node_id=node_id, kind=node.type, meter_id=node.meter_id)
        for node_id, node in graph.nodes.items()
    }


def _format_timestamp(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _consumer_customdata(
    frame_rows: Any, consumer_ids: Iterable[str], measured_col: str
) -> List[Tuple[float, float]]:
    customdata: List[Tuple[float, float]] = []
    for node_id in consumer_ids:
        node_rows = frame_rows[frame_rows["node_id"] == node_id]
        if node_rows.empty:
            measured = 0.0
            true_val = 0.0
        else:
            measured = float(node_rows.iloc[0][measured_col])
            true_val = float(node_rows.iloc[0]["true_flow"])
        customdata.append((measured, true_val))
    return customdata


def build_interactive_network_figure(
    graph: Graph,
    edges_df: Any,
    meters_df: Any,
) -> Any:
    """Create a Plotly figure with a time slider for network flows."""

    try:
        import pandas as pd  # type: ignore
        import plotly.graph_objects as go  # type: ignore
        from plotly.colors import sample_colorscale  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "The interactive network viewer requires pandas and plotly to be installed."
        ) from exc

    if edges_df is None or meters_df is None:
        raise ValueError("Both edges_df and meters_df are required to build the figure.")

    edges_df = pd.DataFrame(edges_df)
    meters_df = pd.DataFrame(meters_df)
    if edges_df.empty or meters_df.empty:
        raise ValueError("Cannot build interactive figure without flow data.")

    positions = _compute_positions(graph)
    if not positions:
        raise ValueError("Could not compute layout positions for the network graph.")

    edge_ids: List[str] = list(graph.edges.keys())
    node_info = _collect_node_info(graph)
    unique_times = sorted(pd.unique(edges_df["time"]))
    if not unique_times:
        raise ValueError("No timestamps available for visualisation.")

    max_edge_flow = float(edges_df["true_flow"].abs().max())
    consumer_rows = meters_df[meters_df["kind"] == "consumer"]
    measured_col = "measured_flow" if "measured_flow" in consumer_rows else "true_flow"
    max_consumer_flow = (
        float(consumer_rows[measured_col].abs().max()) if not consumer_rows.empty else 0.0
    )

    def colour_for_flow(flow: float) -> str:
        norm = _normalise(abs(flow), max_edge_flow)
        return sample_colorscale("Turbo", [norm])[0]

    def build_edge_trace(edge_id: str, flow: float) -> Any:
        edge = graph.edges[edge_id]
        x0, y0 = positions[edge.u]
        x1, y1 = positions[edge.v]
        width = 1.0 + 6.0 * _normalise(abs(flow), max_edge_flow)
        colour = colour_for_flow(flow)
        return go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=colour, width=width),
            hovertemplate=(
                "Edge %s<br>True flow: %.2f<extra></extra>" % (edge_id, flow)
            ),
            showlegend=False,
        )

    consumer_ids: List[str] = [
        nid for nid, info in node_info.items() if info.kind == "consumer" and nid in positions
    ]

    def build_consumer_trace(time_value: Any) -> Any:
        frame_rows = consumer_rows[consumer_rows["time"] == time_value]
        x_vals: List[float] = []
        y_vals: List[float] = []
        texts: List[str] = []
        sizes: List[float] = []
        colours: List[str] = []
        for node_id in consumer_ids:
            info = node_info[node_id]
            x, y = positions[node_id]
            node_rows = frame_rows[frame_rows["node_id"] == node_id]
            if node_rows.empty:
                measured = 0.0
                true_val = 0.0
            else:
                measured = float(node_rows.iloc[0][measured_col])
                true_val = float(node_rows.iloc[0]["true_flow"])
            norm_measured = _normalise(abs(measured), max_consumer_flow)
            size = 12.0 + 18.0 * norm_measured
            colour = sample_colorscale("Teal", [norm_measured])[0]
            label = info.meter_id or node_id
            texts.append(label)
            x_vals.append(x)
            y_vals.append(y)
            sizes.append(size)
            colours.append(colour)
        customdata = _consumer_customdata(frame_rows, consumer_ids, measured_col)
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(size=sizes, color=colours, line=dict(width=1, color="#2f2f2f")),
            hovertemplate=(
                "Consumer %{text}<br>Measured flow: %{customdata[0]:.2f}<br>"
                "True flow: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=customdata,
            showlegend=False,
        )

    source_ids: List[str] = [
        nid for nid, info in node_info.items() if info.kind == "source" and nid in positions
    ]

    def build_source_trace(time_value: Any) -> Any:
        source_rows = meters_df[(meters_df["kind"] == "source") & (meters_df["time"] == time_value)]
        x_vals: List[float] = []
        y_vals: List[float] = []
        texts: List[str] = []
        customdata: List[Tuple[float, float]] = []
        for node_id in source_ids:
            info = node_info[node_id]
            node_rows = source_rows[source_rows["node_id"] == node_id]
            measured = (
                float(node_rows.iloc[0]["measured_flow"])
                if not node_rows.empty and "measured_flow" in node_rows
                else float(node_rows.iloc[0]["true_flow"]) if not node_rows.empty else 0.0
            )
            true_val = float(node_rows.iloc[0]["true_flow"]) if not node_rows.empty else 0.0
            x, y = positions[node_id]
            x_vals.append(x)
            y_vals.append(y)
            texts.append(info.meter_id or node_id)
            customdata.append((measured, true_val))
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            marker=dict(symbol="square", size=16, color="#ffb703", line=dict(width=1.2, color="#2f2f2f")),
            text=texts,
            textposition="bottom center",
            hovertemplate=(
                "Source %{text}<br>Measured flow: %{customdata[0]:.2f}<br>"
                "True flow: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=customdata,
            showlegend=False,
        )

    def build_frame(time_value: Any) -> List[Any]:
        time_edges = edges_df[edges_df["time"] == time_value].set_index("edge_id")
        edge_data = [
            build_edge_trace(edge_id, float(time_edges.loc[edge_id]["true_flow"]) if edge_id in time_edges.index else 0.0)
            for edge_id in edge_ids
        ]
        consumers = build_consumer_trace(time_value)
        sources = build_source_trace(time_value)
        return edge_data + [consumers, sources]

    frames = []
    for ts in unique_times:
        frame_data = build_frame(ts)
        frames.append(go.Frame(name=_format_timestamp(ts), data=frame_data))

    initial_data = frames[0].data
    fig = go.Figure(data=initial_data, frames=frames)

    slider_steps = [
        {
            "args": [
                [frame.name],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
            ],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]

    fig.update_layout(
        title="Interactive Network Flow Viewer",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        sliders=[
            {
                "active": 0,
                "pad": {"t": 50},
                "steps": slider_steps,
                "x": 0.1,
                "len": 0.8,
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 200},
                            },
                        ],
                        "label": "▶ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸ Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 35},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0.0,
            }
        ],
        margin=dict(l=40, r=40, t=80, b=80),
        template="plotly_white",
    )

    fig.update_layout(showlegend=False)
    return fig


__all__ = ["build_interactive_network_figure"]
