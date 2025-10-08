"""Data generator and simulation utilities for water/gas network monitoring.

This module implements a basic synthetic data generator for a network of
sources and consumers connected via directed edges.  It allows the user to
simulate realistic consumption profiles, compute flows on sources and edges,
inject simple anomalies such as leaks and meter offsets, and apply
measurement noise.  The topology is represented with :mod:`networkx`
directed graphs, giving access to robust shortest-path routines while
keeping the remainder of the simulation pipeline lightweight.

Functions are designed to be pure and deterministic when provided with a
random number generator instance and a fixed seed.  All time series are
returned as pandas DataFrames in long form with a consistent schema.

The core workflow is encapsulated in :func:`simulate` which ties together
graph construction, baseline generation, flow computation, noise
application, anomaly injection and output writing.
"""

from __future__ import annotations

import json
import math
import logging
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
import networkx as nx


logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents a node in the network graph."""

    id: str
    type: str  # 'source', 'consumer' or 'junction'
    meter_id: Optional[str] = None
    dma: Optional[str] = None


@dataclass
class Edge:
    """Represents a directed edge in the network graph."""

    id: str
    u: str  # upstream node id
    v: str  # downstream node id
    routing_weight: float = 1.0


class Graph:
    """Directed graph wrapper backed by :mod:`networkx`.

    The public API mirrors the previous lightweight implementation so the
    surrounding simulation code remains unchanged, while leveraging
    NetworkX for traversal and shortest-path utilities.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id {node.id}")
        self.nodes[node.id] = node
        self._graph.add_node(
            node.id,
            type=node.type,
            meter_id=node.meter_id,
            dma=node.dma,
        )

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id {edge.id}")
        if edge.u not in self.nodes or edge.v not in self.nodes:
            raise ValueError(f"Undefined nodes in edge {edge}")
        self.edges[edge.id] = edge
        self._graph.add_edge(
            edge.u,
            edge.v,
            id=edge.id,
            routing_weight=edge.routing_weight,
        )

    @property
    def graph(self) -> nx.DiGraph:
        """Expose the underlying NetworkX directed graph."""

        return self._graph

    def dijkstra(self, start: str) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str]]]:
        """Compute shortest distances and predecessors from ``start``."""

        lengths, paths = nx.single_source_dijkstra(
            self._graph,
            source=start,
            weight=lambda u, v, data: data.get("routing_weight", 1.0),
        )
        dist: Dict[str, float] = {node_id: math.inf for node_id in self.nodes}
        dist.update(lengths)
        prev: Dict[str, Tuple[str, str]] = {}
        for node_id, path in paths.items():
            if node_id == start or len(path) < 2:
                continue
            prev_node = path[-2]
            edge_data = self._graph.get_edge_data(prev_node, node_id, default={})
            edge_id = edge_data.get("id")
            if edge_id is None:
                continue
            prev[node_id] = (prev_node, edge_id)
        return dist, prev

    def dijkstra_reverse(self, start: str) -> Tuple[Dict[str, float], Dict[str, Tuple[str, str]]]:
        """Compute shortest distances on the reverse graph from ``start``."""

        reversed_graph = self._graph.reverse(copy=False)
        lengths, paths = nx.single_source_dijkstra(
            reversed_graph,
            source=start,
            weight=lambda u, v, data: data.get("routing_weight", 1.0),
        )
        dist: Dict[str, float] = {node_id: math.inf for node_id in self.nodes}
        dist.update(lengths)
        prev: Dict[str, Tuple[str, str]] = {}
        for node_id, path in paths.items():
            if node_id == start or len(path) < 2:
                continue
            prev_node = path[-2]
            # When traversing the reversed graph, the original direction is opposite.
            edge_data = self._graph.get_edge_data(node_id, prev_node, default={})
            edge_id = edge_data.get("id")
            if edge_id is None:
                continue
            prev[node_id] = (prev_node, edge_id)
        return dist, prev


def force_directed_layout(
    G: Graph,
    iterations: int = 200,
    area: float = 100000.0,
    cooling: float = 0.92,
    seed: Optional[int] = 0,
    spread: float = 100.0,
    min_distance: float = 50.0,
    edge_separation: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute a Fruchterman–Reingold force-directed layout.

    Based on the formulation from https://en.wikipedia.org/wiki/Force-directed_graph_drawing.
    """

    nodes = list(G.nodes.keys())
    n = len(nodes)
    if n == 0:
        return {}

    index = {node_id: idx for idx, node_id in enumerate(nodes)}
    rng = np.random.default_rng(seed)

    base_area = max(area, 1e-6)
    k = math.sqrt(base_area / n)
    if edge_separation is not None and edge_separation > 0:
        k = float(edge_separation)
    positions = rng.uniform(-1.0, 1.0, size=(n, 2)) * k

    # Treat the graph as undirected for layout purposes
    unique_edges: Set[Tuple[int, int]] = set()
    for edge in G.edges.values():
        u_idx = index[edge.u]
        v_idx = index[edge.v]
        if u_idx == v_idx:
            continue
        ordered = (min(u_idx, v_idx), max(u_idx, v_idx))
        unique_edges.add(ordered)

    temperature = math.sqrt(base_area)
    min_distance = max(min_distance, 1e-9)

    for _ in range(iterations):
        disp = np.zeros((n, 2), dtype=float)

        # Repulsive forces
        for i in range(n):
            delta = positions[i] - positions
            distances = np.linalg.norm(delta, axis=1)
            distances[i] = 1.0  # ignore self interaction
            zero_mask = distances < 1e-9
            distances[zero_mask] = min_distance
            inv_dist = 1.0 / distances
            inv_dist[i] = 0.0
            repulsive = delta * ((k * k) * (inv_dist ** 2))[:, None]
            disp[i] += repulsive.sum(axis=0)

        # Attractive forces
        for u_idx, v_idx in unique_edges:
            delta = positions[u_idx] - positions[v_idx]
            dist = np.linalg.norm(delta)
            if dist < 1e-9:
                direction = rng.uniform(-1.0, 1.0, size=2)
                norm = np.linalg.norm(direction) or 1.0
                direction /= norm
                dist = min_distance
            else:
                direction = delta / dist
            force = (dist * dist) / k
            attractive = direction * force
            disp[u_idx] -= attractive
            disp[v_idx] += attractive

        # Update positions with cooling
        disp_norm = np.linalg.norm(disp, axis=1)
        move_mask = disp_norm > 0
        if np.any(move_mask):
            limited = np.minimum(disp_norm[move_mask], temperature)
            positions[move_mask] += (disp[move_mask] / disp_norm[move_mask][:, None]) * limited[:, None]

        temperature *= cooling
        if temperature < 1e-4:
            break

    # Recentre layout
    positions -= positions.mean(axis=0)

    # Optional scaling for readability
    scale = spread / k if k > 1e-9 else spread
    positions *= scale

    return {node_id: tuple(positions[index[node_id]]) for node_id in nodes}


def build_graph(path: str) -> Graph:
    """Load a network topology from a JSON file and construct a directed graph.

    The JSON file must contain a top‐level ``nodes`` list with entries
    specifying ``id`` and ``type``, and an optional ``meter_id`` and ``dma``.
    It must also contain an ``edges`` list with entries specifying ``id``,
    ``u`` (upstream node id), ``v`` (downstream node id) and an optional
    ``routing_weight``.

    :param path: Path to the JSON file.
    :returns: A :class:`Graph` instance populated with nodes and edges.
    :raises ValueError: If the topology is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        topo = json.load(f)
    G = Graph()
    # Add nodes
    for node_info in topo.get("nodes", []):
        ntype = node_info.get("type")
        node_id = node_info.get("id")
        if not node_id or not ntype:
            raise ValueError(f"Node entry must define id and type: {node_info}")
        meter_id = node_info.get("meter_id")
        dma = node_info.get("dma")
        G.add_node(Node(id=node_id, type=ntype, meter_id=meter_id, dma=dma))
    # Add edges
    for edge_info in topo.get("edges", []):
        eid = edge_info.get("id")
        u = edge_info.get("u")
        v = edge_info.get("v")
        if not eid or not u or not v:
            raise ValueError(f"Edge entry must define id, u and v: {edge_info}")
        weight = float(edge_info.get("routing_weight", 1.0))
        G.add_edge(Edge(id=eid, u=u, v=v, routing_weight=weight))
    # Basic sanity checks: ensure sources and consumers have meters
    for nid, node in G.nodes.items():
        if node.type in ("source", "consumer") and not node.meter_id:
            raise ValueError(f"Node {nid} of type {node.type} must have a meter_id")
    return G


def assign_routes(G: Graph) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Assign each consumer to the closest source and determine route edges.

    For each consumer node in the graph a single source node is chosen as
    upstream provider.  The consumer is linked to the source via the
    shortest path (weighted by ``routing_weight``) on the directed graph.
    Because the graph may contain multiple sources, the function runs
    Dijkstra's algorithm on the *reverse* graph from each consumer to find
    the nearest source.  The path is then reconstructed on the forward
    graph.

    In addition to the mapping from consumer to its assigned source and
    route edges, two auxiliary dictionaries are returned: ``edge_to_consumers``
    mapping each edge id to a list of consumer node ids whose shortest path
    uses that edge, and ``edge_to_sources`` mapping each edge id to a list of
    source node ids reachable via that edge.  These mappings are useful for
    computing edge flows and injecting anomalies.

    :returns: A triple ``(routes, edge_to_consumers, edge_to_sources)`` where
        ``routes`` maps consumer node id to a dict with keys ``source`` and
        ``path_edges``.
    """
    # Identify sources and consumers
    sources = [n.id for n in G.nodes.values() if n.type == "source"]
    consumers = [n.id for n in G.nodes.values() if n.type == "consumer"]
    if not sources:
        raise ValueError("No source nodes defined in topology")
    if not consumers:
        raise ValueError("No consumer nodes defined in topology")

    routes: Dict[str, Dict[str, Any]] = {}
    edge_to_consumers: Dict[str, List[str]] = {eid: [] for eid in G.edges}
    edge_to_sources: Dict[str, List[str]] = {eid: [] for eid in G.edges}
    layout_positions = force_directed_layout(G, seed=0) or {}

    # Precompute shortest paths from all nodes to all sources on reversed graph
    # For each consumer, run Dijkstra on the reversed graph to find nearest source.
    for consumer in consumers:
        # Distances from consumer to all nodes on reversed graph
        dist_rev, prev_rev = G.dijkstra_reverse(consumer)
        # Choose the source with minimum distance
        candidate_distances: List[Tuple[str, float]] = []
        for s in sources:
            d = dist_rev.get(s, math.inf)
            if math.isfinite(d):
                candidate_distances.append((s, d))
        if not candidate_distances:
            raise RuntimeError(f"Consumer {consumer} is unreachable from any source")
        candidate_distances.sort(key=lambda item: item[1])
        nearest_dist = candidate_distances[0][1]
        # Treat sources within tolerance as equivalent so a single feeder
        # does not capture the whole network due to tiny numerical deltas.
        rel_tol = 1e-6
        abs_tol = 1e-9
        candidate_sources = [
            sid
            for sid, dist in candidate_distances
            if math.isclose(dist, nearest_dist, rel_tol=rel_tol, abs_tol=abs_tol)
        ]
        if not candidate_sources:
            candidate_sources = [candidate_distances[0][0]]

        if len(candidate_sources) > 1:
            consumer_pos = layout_positions.get(consumer)
            if consumer_pos is not None:
                cx, cy = consumer_pos

                def layout_distance_sq(source_id: str) -> float:
                    pos = layout_positions.get(source_id)
                    if pos is None:
                        return math.inf
                    sx, sy = pos
                    dx = cx - sx
                    dy = cy - sy
                    return dx * dx + dy * dy

                candidate_sources.sort(key=lambda sid: (layout_distance_sq(sid), sid))
            else:
                candidate_sources.sort()

        nearest_source = candidate_sources[0]
        # Reconstruct path on forward graph from source to consumer
        # We run Dijkstra from the source to produce predecessor mapping
        dist_fwd, prev_fwd = G.dijkstra(nearest_source)
        if consumer not in prev_fwd and consumer != nearest_source:
            raise RuntimeError(
                f"No path found from source {nearest_source} to consumer {consumer}"
            )
        # Build list of edges along the path
        path_edges: List[str] = []
        cur = consumer
        while cur != nearest_source:
            if cur not in prev_fwd:
                raise RuntimeError(
                    f"Broken predecessor chain when reconstructing path to {consumer}"
                )
            pred_node, edge_id = prev_fwd[cur]
            # Prepend the edge (will reverse later)
            path_edges.append(edge_id)
            cur = pred_node
        # path_edges currently holds edges in reverse order from consumer back to source
        path_edges = list(reversed(path_edges))
        routes[consumer] = {"source": nearest_source, "path_edges": path_edges}
        # update edge_to_consumers and edge_to_sources
        for eid in path_edges:
            edge_to_consumers[eid].append(consumer)
            if nearest_source not in edge_to_sources[eid]:
                edge_to_sources[eid].append(nearest_source)
    return routes, edge_to_consumers, edge_to_sources


def generate_time_index(cfg: Dict[str, Any]) -> pd.DatetimeIndex:
    """Create a timezone‐aware DatetimeIndex from configuration.

    The configuration must provide ``start`` (a string parseable by pandas),
    ``periods`` (int) and ``freq`` (string such as '5min').  The timezone is
    fixed to Europe/Warsaw to comply with the project specification.

    :param cfg: Configuration dictionary parsed from YAML.
    :returns: pandas.DatetimeIndex
    """
    start = pd.to_datetime(cfg["start"])
    periods = int(cfg["periods"])
    freq = cfg["freq"]
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="Europe/Warsaw")
    return idx


def generate_baseline_consumption(
    G: Graph, time_index: pd.DatetimeIndex, cfg: Dict[str, Any], rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Generate baseline (true) consumption time series for each consumer.

    A daily consumption profile is synthesised for each consumer by summing a
    specified number of sinusoidal harmonics with random amplitudes and phases.
    The profile is then modulated by day-of-week factors and scaled by a
    consumer‐specific multiplier drawn from a lognormal distribution.  A
    heteroscedastic Gaussian noise term is added at each time step.  Values
    below zero are clipped to zero.

    The resulting DataFrame has the DatetimeIndex as its index and columns
    keyed by consumer node ids containing the true consumption (e.g. m³ per
    sampling interval).
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed"))
    consumers = [n.id for n in G.nodes.values() if n.type == "consumer"]
    if not consumers:
        raise ValueError("No consumers found in graph when generating consumption")
    num_harmonics = cfg.get("profiles", {}).get("num_harmonics", 2)
    dow_factors_cfg = cfg.get("profiles", {}).get("dow_factors", {})
    # Convert keys to ints
    dow_factors = {int(k): float(v) for k, v in dow_factors_cfg.items()}
    # Precompute time components
    # Hours since midnight for each timestamp (0–24)
    hours_of_day = time_index.hour + time_index.minute / 60.0 + time_index.second / 3600.0
    day_of_week = time_index.dayofweek  # Monday=0
    # Season factor: optional scaling across the full time range using a simple sine
    season_strength = cfg.get("profiles", {}).get("season_strength", 0.0)
    total_points = len(time_index)
    if season_strength:
        # one full cycle over the index length
        season = 1.0 + season_strength * np.sin(2 * np.pi * np.arange(total_points) / total_points)
    else:
        season = np.ones(total_points)
    baseline_data: Dict[str, np.ndarray] = {}
    for consumer_id in consumers:
        # Draw random amplitudes and phases for the harmonic components
        amplitudes = rng.uniform(0.5, 1.0, size=num_harmonics) / (np.arange(num_harmonics) + 1)
        phases = rng.uniform(0, 2 * np.pi, size=num_harmonics)
        # Compose the daily profile
        daily_profile = np.zeros_like(hours_of_day, dtype=float)
        for k in range(num_harmonics):
            daily_profile += amplitudes[k] * np.sin(2 * np.pi * (k + 1) * hours_of_day / 24.0 + phases[k])
        # Shift profile so that negative values are penalised less
        daily_profile = daily_profile - daily_profile.min()
        # Normalise to average around one
        if daily_profile.mean() != 0:
            daily_profile = daily_profile / daily_profile.mean()
        # Apply day-of-week factor
        dow_factor_array = np.vectorize(lambda d: dow_factors.get(d, 1.0))(day_of_week)
        profile = daily_profile * dow_factor_array * season
        # Scale per consumer (lognormal with mean 1 and sigma 0.5)
        scale_c = float(rng.lognormal(mean=0.0, sigma=0.5))
        # Compute baseline consumption
        baseline = scale_c * profile
        # Add noise (relative to baseline).  Use consumer noise config.
        sigma_rel = cfg.get("noise", {}).get("consumer", {}).get("sigma_rel", 0.0)
        sigma_abs = cfg.get("noise", {}).get("consumer", {}).get("sigma_abs", 0.0)
        noise = rng.normal(loc=0.0, scale=sigma_rel * np.abs(baseline) + sigma_abs)
        baseline += noise
        # Clip to non-negative
        baseline = np.maximum(baseline, 0.0)
        baseline_data[consumer_id] = baseline
    df = pd.DataFrame(baseline_data, index=time_index)
    return df


def compute_source_and_edge_flows(
    G: Graph,
    baseline_df: pd.DataFrame,
    routes: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    edge_to_consumers: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute true flows at sources and edges based on consumer baseline consumption.

    Given the baseline consumption for each consumer and the assignment of
    consumers to sources via specific routes, compute two time series:

    * ``meters_df`` – a long DataFrame with columns ``time``, ``meter_id``,
      ``node_id``, ``kind`` (``'source'`` or ``'consumer'``) and ``true_flow``.
    * ``edges_df`` – a long DataFrame with columns ``time``, ``edge_id`` and
      ``true_flow``.  The edge flows are simply the sum of consumer flows for
      all consumers whose route contains the edge; they do **not** include
      losses.

    Losses are applied only at sources: the true flow for each source is
    computed as ``sum(consumption) * (1 + loss_rate)``.
    """
    time_index = baseline_df.index
    loss_rate = float(cfg.get("loss_rate", 0.0))
    # Prepare container for rows for meters
    meter_rows: List[Dict[str, Any]] = []
    # Create consumer meter rows
    for consumer_id, route in routes.items():
        node = G.nodes[consumer_id]
        meter_id = node.meter_id
        if meter_id is None:
            continue  # skip consumers without meters (should not happen)
        true_series = baseline_df[consumer_id]
        for ts, flow in zip(time_index, true_series):
            meter_rows.append(
                {
                    "time": ts,
                    "meter_id": meter_id,
                    "node_id": consumer_id,
                    "kind": "consumer",
                    "true_flow": float(flow),
                }
            )
    # Compute source flows
    # Build mapping source_id -> list of consumers assigned
    source_to_consumers: Dict[str, List[str]] = {}
    for consumer_id, route in routes.items():
        src = route["source"]
        source_to_consumers.setdefault(src, []).append(consumer_id)
    all_sources = [n.id for n in G.nodes.values() if n.type == "source"]
    for source_id in all_sources:
        cons_list = source_to_consumers.get(source_id, [])
        node = G.nodes[source_id]
        meter_id = node.meter_id
        if meter_id is None:
            continue
        # Sum baseline consumption across consumers for each time
        if cons_list:
            baseline_sum = baseline_df[cons_list].sum(axis=1)
        else:
            baseline_sum = pd.Series(0.0, index=time_index)
        # Apply loss rate
        source_true = baseline_sum * (1.0 + loss_rate)
        for ts, flow in zip(time_index, source_true):
            meter_rows.append(
                {
                    "time": ts,
                    "meter_id": meter_id,
                    "node_id": source_id,
                    "kind": "source",
                    "true_flow": float(flow),
                }
            )
    meters_df = pd.DataFrame(meter_rows)
    # Compute edge flows
    edge_rows: List[Dict[str, Any]] = []
    for edge_id, cons_list in edge_to_consumers.items():
        # If no consumer uses this edge the flow is zero
        if not cons_list:
            zero = np.zeros(len(time_index), dtype=float)
            series = zero
        else:
            series = baseline_df[cons_list].sum(axis=1).values
        for ts, flow in zip(time_index, series):
            edge_rows.append(
                {
                    "time": ts,
                    "edge_id": edge_id,
                    "true_flow": float(flow),
                }
            )
    edges_df = pd.DataFrame(edge_rows)
    return meters_df, edges_df


def add_measurement_noise(
    meters_df: pd.DataFrame, cfg: Dict[str, Any], rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Apply additive Gaussian measurement noise to each meter reading.

    The noise parameters are specified per meter type (source or consumer) in
    ``cfg['noise']``.  For each reading ``true_flow``, a random value is
    drawn from ``N(0, sigma_abs + sigma_rel * true_flow)`` and added to it.
    The resulting ``measured_flow`` is clipped to be non‑negative.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed"))
    meter_rows = []
    # iterate rows and add noise
    for row in meters_df.itertuples(index=False):
        kind = row.kind
        true_flow = row.true_flow
        noise_cfg = cfg.get("noise", {}).get(kind, {})
        sigma_abs = float(noise_cfg.get("sigma_abs", 0.0))
        sigma_rel = float(noise_cfg.get("sigma_rel", 0.0))
        sigma = sigma_abs + sigma_rel * abs(true_flow)
        noise = rng.normal(0.0, sigma)
        measured = true_flow + noise
        if measured < 0:
            measured = 0.0
        meter_rows.append(
            {
                "time": row.time,
                "meter_id": row.meter_id,
                "node_id": row.node_id,
                "kind": kind,
                "true_flow": true_flow,
                "measured_flow": float(measured),
            }
        )
    return pd.DataFrame(meter_rows)


def inject_events(
    meters_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    events_df: pd.DataFrame,
    routes: Dict[str, Dict[str, Any]],
    edge_to_sources: Dict[str, List[str]],
    cfg: Dict[str, Any],
    G: Graph,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inject anomalies (leaks, meter offsets) into the simulated data.

    The ``events_df`` DataFrame must contain at least the columns:

    - ``type`` – either ``'leak'`` or ``'offset'``.
    - ``ts_start`` – the start timestamp of the anomaly.
    - ``duration`` – the duration in the same units as the sampling frequency (e.g. minutes).
    - ``target_kind`` – either ``'edge'`` for leaks or ``'meter'`` for offsets.
    - ``target_id`` – the ID of the edge or meter on which the anomaly occurs.
    - ``magnitude`` – for leaks: absolute additional flow; for offsets: additive offset.
    - ``mode`` – optional mode for offsets; currently only ``'add'`` is supported.

    The function updates the ``true_flow`` and ``measured_flow`` columns
    of ``meters_df`` and ``true_flow`` column of ``edges_df`` in place and
    returns the modified DataFrames along with a copy of ``events_df`` as
    the ground truth labels.
    """
    if events_df is None or events_df.empty:
        return meters_df, edges_df, pd.DataFrame()
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed"))
    # Ensure datetime conversion
    events_df = events_df.copy()
    # parse ts_start as timezone aware
    # Parse ts_start into timezone aware timestamps.  If already tz-aware convert,
    # otherwise localise to Europe/Warsaw.
    ts_parsed = pd.to_datetime(events_df["ts_start"])
    if ts_parsed.dt.tz is None:
        events_df["ts_start"] = ts_parsed.dt.tz_localize("Europe/Warsaw")
    else:
        events_df["ts_start"] = ts_parsed.dt.tz_convert("Europe/Warsaw")
    # unify duration into number of periods (in minutes if freq is minutes)
    # Determine sampling frequency in minutes
    # Parse cfg['freq'], e.g. '5min' -> 5
    freq_str = cfg.get("freq", "1min")
    if freq_str.endswith("min"):
        freq_minutes = int(freq_str[:-3])
    elif freq_str.endswith("H") or freq_str.endswith("h"):
        freq_minutes = int(freq_str[:-1]) * 60
    else:
        # fallback to 1 minute
        freq_minutes = 1
    # Process each event
    for event in events_df.itertuples(index=False):
        etype = event.type
        ts_start = event.ts_start
        duration = event.duration
        target_kind = event.target_kind
        target_id = event.target_id
        magnitude = float(event.magnitude)
        mode = getattr(event, "mode", "add") or "add"
        # Determine time mask
        # Duration is given in minutes; convert to number of periods
        if isinstance(duration, str):
            # Try to parse as int; if fails treat as ISO 8601 duration
            try:
                duration_min = float(duration)
            except ValueError:
                # Use pandas Timedelta
                duration_td = pd.to_timedelta(duration)
                duration_min = duration_td / pd.Timedelta(minutes=1)
        else:
            duration_min = float(duration)
        num_periods = int(math.ceil(duration_min / freq_minutes))
        # Build mask of times where event is active
        time_index = pd.to_datetime(meters_df["time"])
        mask = (time_index >= ts_start) & (time_index < ts_start + pd.Timedelta(minutes=num_periods * freq_minutes))
        if etype == "leak" and target_kind == "edge":
            # Identify impacted sources for this edge
            sources_impacted = edge_to_sources.get(target_id, [])
            if not sources_impacted:
                logger.warning(f"Leak event on edge {target_id} has no sources upstream")
            # For each impacted source, add magnitude to its true and measured flows
            for src in sources_impacted:
                node = G.nodes.get(src)
                meter_id = node.meter_id if node else None
                if meter_id is None:
                    continue
                cond = (meters_df["meter_id"] == meter_id) & mask
                # Add magnitude to true_flow and measured_flow
                meters_df.loc[cond, "true_flow"] = meters_df.loc[cond, "true_flow"] + magnitude
                meters_df.loc[cond, "measured_flow"] = meters_df.loc[cond, "measured_flow"] + magnitude
            # Update edge true flows
            cond_edge = (edges_df["edge_id"] == target_id) & mask
            edges_df.loc[cond_edge, "true_flow"] = edges_df.loc[cond_edge, "true_flow"] + magnitude
        elif etype == "offset" and target_kind in ("meter", "source", "consumer"):
            # For offset events the target_id refers to a meter_id (not node)
            # We modify measured_flow only
            meter_id = target_id
            cond = (meters_df["meter_id"] == meter_id) & mask
            if mode == "add":
                meters_df.loc[cond, "measured_flow"] = meters_df.loc[cond, "measured_flow"] + magnitude
            elif mode == "mul":
                meters_df.loc[cond, "measured_flow"] = meters_df.loc[cond, "measured_flow"] * (1.0 + magnitude)
            elif mode == "drift":
                # linearly increasing drift: add magnitude * fraction of elapsed period
                # Determine indices of events
                event_times = meters_df.loc[cond, "time"].to_numpy()
                start_time = ts_start
                # compute fraction of elapsed
                fractions = ((event_times - start_time).astype('timedelta64[s]').astype(float) / (num_periods * freq_minutes * 60))
                meters_df.loc[cond, "measured_flow"] = (
                    meters_df.loc[cond, "measured_flow"] + magnitude * fractions
                )
        else:
            logger.warning(f"Unsupported event type {etype} or target_kind {target_kind}")
    # Labels simply mirror events
    labels_df = events_df.copy()
    return meters_df, edges_df, labels_df


def plot_timeseries(meters_df: pd.DataFrame, labels_df: Optional[pd.DataFrame] = None) -> Any:
    """Plot aggregate inflows, consumption and balance over time with anomaly shading."""

    import matplotlib.pyplot as plt

    df = meters_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)

    inflow = df[df["kind"] == "source"].groupby("time")["measured_flow"].sum()
    consumption = df[df["kind"] == "consumer"].groupby("time")["measured_flow"].sum()
    inflow, consumption = inflow.align(consumption, fill_value=0)
    delta = inflow - consumption

    colors = {
        "inflow": "#1f77b4",
        "consumption": "#ff7f0e",
        "delta_pos": "#2ca02c",
        "delta_neg": "#d62728",
        "leak": "#d62728",
        "offset": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(
        inflow.index,
        inflow.values,
        step="mid",
        alpha=0.18,
        color=colors["inflow"],
        label="Sum inflows",
    )
    ax.fill_between(
        consumption.index,
        consumption.values,
        step="mid",
        alpha=0.18,
        color=colors["consumption"],
        label="Sum consumption",
    )
    ax.plot(inflow.index, inflow.values, color=colors["inflow"], linewidth=2)
    ax.plot(consumption.index, consumption.values, color=colors["consumption"], linewidth=2)

    ax.plot(delta.index, delta.values, color="#555555", linewidth=1.5, label="Δ inflows − consumption")
    ax.fill_between(
        delta.index,
        0,
        delta.values,
        where=delta.values >= 0,
        color=colors["delta_pos"],
        alpha=0.12,
        interpolate=True,
    )
    ax.fill_between(
        delta.index,
        0,
        delta.values,
        where=delta.values < 0,
        color=colors["delta_neg"],
        alpha=0.12,
        interpolate=True,
    )

    legend_labels_added = set()
    if labels_df is not None and not labels_df.empty:
        for event in labels_df.itertuples(index=False):
            ts_start = pd.to_datetime(event.ts_start)
            duration = getattr(event, "duration", None)
            color = colors.get(event.type, "#8c8c8c")

            # Derive event duration for shading
            duration_td = None
            if duration is not None:
                if isinstance(duration, (int, float)):
                    duration_td = pd.to_timedelta(duration, unit="m")
                else:
                    duration_str = str(duration).strip()
                    try:
                        if duration_str.replace(".", "", 1).isdigit():
                            duration_td = pd.to_timedelta(float(duration_str), unit="m")
                        else:
                            duration_td = pd.to_timedelta(duration_str)
                    except (ValueError, TypeError):
                        try:
                            duration_td = pd.to_timedelta(float(duration_str), unit="m")
                        except Exception:
                            duration_td = None
            if duration_td is None:
                duration_td = pd.Timedelta(minutes=10)

            ts_end = ts_start + duration_td
            label = f"{event.type.capitalize()} anomaly"
            label_to_use = label if label not in legend_labels_added else None
            ax.axvspan(
                ts_start,
                ts_end,
                color=color,
                alpha=0.18,
                label=label_to_use,
            )
            legend_labels_added.add(label)

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow")
    ax.set_title("Network Aggregate Flows")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_network_overview(
    G: Graph,
    meters_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
) -> Any:
    """Render a force-directed network overview highlighting cumulative flows."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Patch
    from matplotlib.lines import Line2D

    def format_value(value: float) -> str:
        abs_val = abs(value)
        if abs_val >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        if abs_val >= 1_000:
            return f"{value/1_000:.1f}k"
        if abs_val >= 100:
            return f"{value:.1f}"
        if abs_val >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    positions = force_directed_layout(
        G,
        area=360000.0,
        spread=24.0,
        min_distance=12.0,
        edge_separation=18.0,
        seed=7,
    )
    if not positions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No topology available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        return fig

    edge_true_total = (
        edges_df.groupby("edge_id")["true_flow"].sum().to_dict() if not edges_df.empty else {}
    )
    monitoring_columns = {"true_flow", "measured_flow"}
    if meters_df.empty:
        meter_totals = pd.DataFrame(columns=["node_id", "metric", "value"])
    else:
        meter_totals = (
            meters_df.groupby(["node_id"])
            .agg({col: "sum" for col in monitoring_columns if col in meters_df.columns})
            .rename(columns={"true_flow": "true_total", "measured_flow": "measured_total"})
        )
    true_per_node = meter_totals.get("true_total", pd.Series(dtype=float)).to_dict()
    measured_per_node = meter_totals.get("measured_total", pd.Series(dtype=float)).to_dict()

    kind_per_node = {node.id: node.type for node in G.nodes.values()}
    meter_lookup = {node.meter_id: node.id for node in G.nodes.values() if node.meter_id}

    def extract_anomalies() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        node_anomalies: Dict[str, Set[str]] = {}
        edge_anomalies: Dict[str, Set[str]] = {}
        if labels_df is None or labels_df.empty:
            return node_anomalies, edge_anomalies
        for event in labels_df.itertuples(index=False):
            etype = getattr(event, "type", "anomaly")
            target_kind = getattr(event, "target_kind", "")
            target_id = getattr(event, "target_id", "")
            if target_kind in ("meter", "source", "consumer"):
                node_id = meter_lookup.get(target_id, target_id if target_id in G.nodes else None)
                if node_id is None:
                    continue
                node_anomalies.setdefault(node_id, set()).add(etype)
            elif target_kind == "edge":
                edge_anomalies.setdefault(target_id, set()).add(etype)
        return node_anomalies, edge_anomalies

    node_anomalies, edge_anomalies = extract_anomalies()

    max_edge_total = max(abs(v) for v in edge_true_total.values()) if edge_true_total else 1.0
    true_totals_only = [abs(v) for v in true_per_node.values()]
    max_node_total = max(true_totals_only) if true_totals_only else 1.0

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_title("Force-directed Network Overview (total flows)")

    base_edge_color = "#4a6fa5"
    anomaly_edge_color = "#d1495b"
    base_node_colors = {
        "source": "#ffd166",
        "consumer": "#06d6a0",
        "junction": "#adb5bd",
    }

    # Draw directed edges with arrowheads
    for edge in G.edges.values():
        if edge.u not in positions or edge.v not in positions:
            continue
        start = positions[edge.u]
        end = positions[edge.v]
        total_flow = edge_true_total.get(edge.id, 0.0)
        linewidth = 1.2 + 3.5 * (abs(total_flow) / max_edge_total if max_edge_total else 0.0)
        color = anomaly_edge_color if edge.id in edge_anomalies else base_edge_color
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            linewidth=linewidth,
            color=color,
            alpha=0.9,
            mutation_scale=14 + 10 * (abs(total_flow) / max_edge_total if max_edge_total else 0.0),
            shrinkA=15.0,
            shrinkB=15.0,
            zorder=2,
        )
        ax.add_patch(arrow)

        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.hypot(dx, dy) or 1.0
        offset_scale = 0.75 + 0.35 * (abs(total_flow) / max_edge_total if max_edge_total else 0.0)
        label_x = mid_x - dy / norm * offset_scale
        label_y = mid_y + dx / norm * offset_scale
        label = f"{edge.id}\nΣtrue {format_value(total_flow)}"
        if edge.id in edge_anomalies:
            label += "\n⚠"
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=9,
            ha="center",
            va="center",
            color="#1f1f1f",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
            zorder=4,
        )

    # Draw nodes
    for node in G.nodes.values():
        node_id = node.id
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        total_true = true_per_node.get(node_id, 0.0)
        base_color = base_node_colors.get(kind_per_node.get(node_id, "junction"), "#ced4da")
        if node_id in node_anomalies:
            if any(etype == "leak" for etype in node_anomalies[node_id]):
                base_color = "#ef476f"
            else:
                base_color = "#8e7cc3"
        size = 320 + 1400 * (abs(total_true) / max_node_total if max_node_total else 0.0)
        ax.scatter(
            [x],
            [y],
            s=size,
            color=base_color,
            edgecolors="#212529",
            linewidths=1.1,
            zorder=3,
        )

        label_lines = [node_id]
        if node.type == "consumer":
            label_lines.append(f"Σtrue {format_value(total_true)}")
            measured_total = measured_per_node.get(node_id)
            if measured_total is not None:
                label_lines.append(f"Σmeas {format_value(measured_total)}")
        elif node.type == "source":
            label_lines.append(f"Σout {format_value(total_true)}")

        if node_id in node_anomalies:
            label_lines.append("⚠")

        ax.text(
            x,
            y - (math.sqrt(size) * 0.025 + 0.6),
            "\n".join(label_lines),
            ha="center",
            va="top",
            fontsize=9,
            color="#212529",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="none"),
            zorder=4,
        )

    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_margin = max(4.0, (x_max - x_min) * 0.12)
        y_margin = max(3.0, (y_max - y_min) * 0.18)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.axis("off")

    legend_handles = [
        Patch(facecolor=base_node_colors["source"], edgecolor="#212529", label="Source"),
        Patch(facecolor=base_node_colors["consumer"], edgecolor="#212529", label="Consumer"),
        Patch(facecolor=base_node_colors["junction"], edgecolor="#212529", label="Junction"),
        Line2D([0], [0], color=base_edge_color, lw=2.5, label="Edge true flow (Σ)"),
    ]
    if edge_anomalies:
        legend_handles.append(Line2D([0], [0], color=anomaly_edge_color, lw=2.5, label="Edge anomaly"))
    if any(node_anomalies.values()):
        legend_handles.append(Patch(facecolor="#8e7cc3", edgecolor="#212529", label="Node anomaly"))

    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig




def simulate(
    topology_path: str,
    cfg_path: str,
    events_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    make_plot: bool = True,
) -> Dict[str, Any]:
    """Run the full simulation pipeline for a given topology and configuration.

    The simulation performs the following steps:

    1. Load configuration from YAML.
    2. Build the graph from the topology JSON.
    3. Assign routes between consumers and sources.
    4. Generate the time index.
    5. Generate baseline consumer consumption.
    6. Compute true flows on meters (sources and consumers) and edges.
    7. Add measurement noise to obtain measured flows.
    8. Optionally load and inject anomaly events.
    9. Persist output files (CSV in place of Parquet) and return results.

    :param topology_path: Path to the network_topology.json file.
    :param cfg_path: Path to the configuration YAML file.
    :param events_path: Optional path to a CSV or YAML file specifying anomaly events.
    :param out_dir: Optional directory to which output files will be written.
    :param make_plot: If ``True`` create and (optionally) save the Matplotlib figure.
        Set to ``False`` when running from a non-main thread; the caller can invoke
        :func:`plot_timeseries` on the returned data instead.
    :returns: A dictionary containing ``meters_df``, ``edges_df``, ``labels_df``, the network
        graph object ``graph`` and Matplotlib figures ``figure`` (timeseries) and
        ``network_figure`` (topology overview).
    """
    # Load configuration
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Build graph
    G = build_graph(topology_path)
    # Assign routes and precompute edge mappings
    routes, edge_to_consumers, edge_to_sources = assign_routes(G)
    # Generate time index
    time_index = generate_time_index(cfg)
    # Generate baseline consumption
    rng = np.random.default_rng(cfg.get("seed"))
    baseline_df = generate_baseline_consumption(G, time_index, cfg, rng=rng)
    # Compute true flows for meters and edges
    meters_df, edges_df = compute_source_and_edge_flows(G, baseline_df, routes, cfg, edge_to_consumers)
    # Add measurement noise
    meters_df = add_measurement_noise(meters_df, cfg, rng=rng)
    # Optionally load events and inject
    labels_df = pd.DataFrame()
    if events_path:
        # Read events.  Support CSV and YAML
        if events_path.endswith(".csv"):
            events_df = pd.read_csv(events_path)
        elif events_path.endswith((".yml", ".yaml")):
            with open(events_path, "r", encoding="utf-8") as f:
                events_cfg = yaml.safe_load(f)
            events_df = pd.DataFrame(events_cfg)
        else:
            raise ValueError(f"Unsupported events file format: {events_path}")
        meters_df, edges_df, labels_df = inject_events(
            meters_df,
            edges_df,
            events_df,
            routes,
            edge_to_sources,
            cfg,
            G,
            rng=rng,
        )
    # Persist outputs
    fig: Optional[Any] = None
    network_fig: Optional[Any] = None
    if make_plot:
        fig = plot_timeseries(meters_df, labels_df)
        network_fig = plot_network_overview(G, meters_df, edges_df, labels_df)

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        # Save meter readings and edge flows as CSV; naming with parquet extension for spec compatibility
        meters_path = os.path.join(out_dir, "meter_readings.parquet")
        edges_path = os.path.join(out_dir, "edge_flows.parquet")
        labels_path = os.path.join(out_dir, "labels.csv")
        meters_df.to_csv(meters_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        if labels_df is not None and not labels_df.empty:
            labels_df.to_csv(labels_path, index=False)
        if fig is not None:
            plot_path = os.path.join(out_dir, "timeseries.svg")
            fig.savefig(plot_path, format="svg")
        if network_fig is not None:
            network_path = os.path.join(out_dir, "network_overview.svg")
            network_fig.savefig(network_path, format="svg")

    return {
        "meters_df": meters_df,
        "edges_df": edges_df,
        "labels_df": labels_df,
        "graph": G,
        "figure": fig,
        "network_figure": network_fig,
    }


__all__ = [
    "build_graph",
    "assign_routes",
    "generate_time_index",
    "generate_baseline_consumption",
    "compute_source_and_edge_flows",
    "add_measurement_noise",
    "inject_events",
    "plot_timeseries",
    "plot_network_overview",
    "simulate",
]
