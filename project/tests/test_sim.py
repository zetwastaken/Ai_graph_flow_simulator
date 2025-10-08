"""Tests for the simulation pipeline functions."""

import pandas as pd
import numpy as np

from project.sim.generator import (
    build_graph,
    assign_routes,
    generate_time_index,
    generate_baseline_consumption,
    compute_source_and_edge_flows,
    add_measurement_noise,
    inject_events,
)


def test_baseline_balance_no_noise():
    """Baseline without noise and losses should have zero balance."""
    # Build graph and assign routes
    G = build_graph("project/data/network_topology.json")
    routes, edge_to_consumers, edge_to_sources = assign_routes(G)
    # Create configuration with no losses and no noise
    cfg = {
        "start": "2025-01-01 00:00",
        "periods": 24,
        "freq": "1H",  # hourly resolution for test simplicity
        "seed": 0,
        "loss_rate": 0.0,
        "noise": {
            "source": {"sigma_abs": 0.0, "sigma_rel": 0.0},
            "consumer": {"sigma_abs": 0.0, "sigma_rel": 0.0},
        },
        "profiles": {
            "num_harmonics": 1,
            "dow_factors": {str(i): 1.0 for i in range(7)},
            "season_strength": 0.0,
        },
    }
    idx = generate_time_index(cfg)
    rng = np.random.default_rng(cfg["seed"])
    baseline_df = generate_baseline_consumption(G, idx, cfg, rng=rng)
    meters_df, edges_df = compute_source_and_edge_flows(G, baseline_df, routes, cfg, edge_to_consumers)
    meters_df = add_measurement_noise(meters_df, cfg, rng=rng)
    # Compute balance for each time: inflow - consumption
    df = meters_df
    inflow = df[df["kind"] == "source"].groupby("time")["measured_flow"].sum()
    consumption = df[df["kind"] == "consumer"].groupby("time")["measured_flow"].sum()
    delta = inflow - consumption
    # All deltas should be nearly zero
    assert np.allclose(delta.values, 0.0, atol=1e-8)


def test_leak_increases_delta_by_magnitude():
    """Injecting a leak should increase the inflow minus consumption by its magnitude."""
    # Build graph and assign routes
    G = build_graph("project/data/network_topology.json")
    routes, edge_to_consumers, edge_to_sources = assign_routes(G)
    # Short config for test
    cfg = {
        "start": "2025-01-01 00:00",
        "periods": 12,
        "freq": "5min",
        "seed": 42,
        "loss_rate": 0.0,
        "noise": {
            "source": {"sigma_abs": 0.0, "sigma_rel": 0.0},
            "consumer": {"sigma_abs": 0.0, "sigma_rel": 0.0},
        },
        "profiles": {
            "num_harmonics": 1,
            "dow_factors": {str(i): 1.0 for i in range(7)},
            "season_strength": 0.0,
        },
    }
    idx = generate_time_index(cfg)
    rng = np.random.default_rng(cfg["seed"])
    baseline_df = generate_baseline_consumption(G, idx, cfg, rng=rng)
    meters_df, edges_df = compute_source_and_edge_flows(G, baseline_df, routes, cfg, edge_to_consumers)
    meters_df = add_measurement_noise(meters_df, cfg, rng=rng)
    # Compute baseline delta at each time
    inflow = meters_df[meters_df["kind"] == "source"].groupby("time")["measured_flow"].sum()
    consumption = meters_df[meters_df["kind"] == "consumer"].groupby("time")["measured_flow"].sum()
    baseline_delta = (inflow - consumption).to_dict()
    # Define a leak event on edge e2
    event_start = idx[3]  # choose the 4th timestamp
    magnitude = 5.0
    events_df = pd.DataFrame([
        {
            "id": "ev1",
            "type": "leak",
            "ts_start": str(event_start),
            "duration": 10,  # minutes
            "target_kind": "edge",
            "target_id": "e2",
            "magnitude": magnitude,
            "mode": "const",
        }
    ])
    # Inject event
    meters_df2, edges_df2, labels_df = inject_events(
        meters_df.copy(), edges_df.copy(), events_df, routes, edge_to_sources, cfg, G, rng=rng
    )
    # Compute delta after injection
    inflow2 = meters_df2[meters_df2["kind"] == "source"].groupby("time")["measured_flow"].sum()
    consumption2 = meters_df2[meters_df2["kind"] == "consumer"].groupby("time")["measured_flow"].sum()
    delta2 = inflow2 - consumption2
    # For times during the event, delta should increase by magnitude
    for ts in idx:
        if ts >= event_start and ts < event_start + pd.Timedelta(minutes=10):
            assert np.isclose(delta2.loc[ts] - baseline_delta[ts], magnitude, atol=1e-8), (
                f"Delta should increase by magnitude at {ts}"
            )
        else:
            assert np.isclose(delta2.loc[ts], baseline_delta[ts], atol=1e-8), (
                f"Delta should remain unchanged outside event period at {ts}"
            )