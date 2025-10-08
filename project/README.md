# Data Simulator for Network Monitoring

This repository contains an MVP implementation of a synthetic data
generator for water/gas network monitoring.  The goal is to provide
repeatable time‑series data for research and prototyping of leak and
meter fault detection algorithms without requiring access to real
infrastructure.  The design draws directly from the specification
outlined in the problem statement and emphasises modularity and future
extensibility.

## Quick start

To run the simulator on the bundled sample topology and configuration
without persisting outputs, execute:

```bash
python -m project.cli simulate \
  --topology project/data/network_topology.json \
  --config project/config.yml
```

To store the generated meter readings, edge flows, event labels and
time‑series plot to a directory (e.g. `out/`), provide the `--out`
parameter:

```bash
python -m project.cli simulate \
  --topology project/data/network_topology.json \
  --config project/config.yml \
  --out out
```

The simulator reads its inputs from YAML/JSON files and produces
CSV files with a `.parquet` extension for compatibility with the
original specification.  The high‑level workflow is described in
`project/sim/generator.py` and consists of:

1. Building a directed graph from the network topology.
2. Assigning each consumer to the nearest source via a shortest path.
3. Synthesising realistic baseline consumption profiles with daily
   harmonics, day‑of‑week modulation, random scaling and noise.
4. Computing true flows at sources and along edges.
5. Applying measurement noise to obtain measured flows.
6. Optionally injecting anomalies such as leaks and meter offsets.
7. Persisting the resulting data frames and drawing a summary plot.

## Project structure

```
project/
  config.yml                  # Sample simulator configuration
  data/
    network_topology.json     # Sample network with one source and two consumers
  sim/
    generator.py              # Core implementation of the data generator
    anomalies.py              # (placeholder) future event injection logic
  pipeline/
    feature_engineering.py    # (placeholder) future feature engineering utilities
  detectors/                  # (placeholder) future detection algorithms
  diagnostics/                # (placeholder) future diagnostic reasoning
  evaluation/                 # (placeholder) future performance metrics
  viz/
    dashboard.py              # CLI for running a simulation and plotting results
  cli.py                      # Command line entrypoint for the simulator
  tests/
    test_network.py           # Unit tests for graph loading and routing
    test_sim.py               # Unit tests for baseline and anomaly injection
  README.md                   # This file
```

Placeholders (`detectors/`, `diagnostics/`, `evaluation/`, etc.) mark where
additional functionality will be integrated in subsequent milestones
without cluttering the current MVP.  The simulator is intentionally
self‑contained in pure Python and does not rely on external graph or
Parquet libraries, which simplifies execution in constrained
environments.

## Writing custom topologies and events

To simulate a different network, provide a `network_topology.json` with
objects under `nodes` and `edges`.  Sources and consumers must define a
`meter_id`.  Edges are directed from `u` to `v` and can optionally
specify a `routing_weight` used in shortest path calculations.

Event definitions may be supplied in a CSV file with columns like:

| id  | type   | ts_start            | duration | target_kind | target_id | magnitude | mode  |
|-----|--------|---------------------|----------|-------------|-----------|-----------|-------|
| ev1 | leak   | 2025-01-01 01:00:00 | 60       | edge        | e2        | 10.0      | const |
| ev2 | offset | 2025-01-01 03:00:00 | 30       | meter       | meter_c1  | 3.0       | add   |

A *leak* on an edge increases the flow measured at upstream sources but
does not affect any consumer meter, thereby increasing the net inflow.
A *meter offset* event biases the measured flow of a specific meter
without changing the true flow.

## Running tests

While `pytest` is not available in this environment, the unit tests can
be executed directly using the Python interpreter.  Navigate to the
repository root and run:

```bash
python -m project.tests.test_network
python -m project.tests.test_sim
```

These tests verify that the sample topology loads correctly, that the
routing logic is deterministic, and that the baseline and leak
injection behave as expected.

## Limitations and future work

The current MVP focuses solely on data generation and simple
visualisation.  It does not yet support real Parquet output, live
streaming, or advanced anomaly detection.  Future iterations will
populate the placeholder modules with feature engineering routines,
anomaly detection algorithms, diagnostic reasoning and performance
evaluation.  Integration with time‑series databases, dashboards and
APIs is also planned.