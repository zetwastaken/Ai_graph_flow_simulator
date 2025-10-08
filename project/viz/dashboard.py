"""Simple visualisation wrapper for the simulator output.

Currently this module exposes a convenience function to run a
simulation and display the aggregate timeseries plot.  In the future
this may be extended into an interactive dashboard using e.g. Plotly
Dash or Bokeh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import simulate with fallback when executed outside package context
try:
    from ..sim.generator import simulate  # type: ignore
except Exception:
    base_dir_for_imports = Path(__file__).resolve().parents[1]
    if str(base_dir_for_imports) not in sys.path:
        sys.path.insert(0, str(base_dir_for_imports))
    from sim.generator import simulate  # type: ignore


def main() -> None:
    """Entry point for generating and displaying a simulation plot.

    This CLI accepts arguments for the topology, config and events files.  It
    runs the simulation and shows the resulting plot.  Outputs are not
    persisted by default.
    """
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run simulation and display plot")
    parser.add_argument("--topology", default=str(base_dir / "data" / "network_topology.json"), help="Path to network topology JSON")
    parser.add_argument("--config", default=str(base_dir / "config.yml"), help="Path to configuration YAML")
    parser.add_argument("--events", default=None, help="Path to events CSV/YAML (optional)")
    args = parser.parse_args()
    result = simulate(args.topology, args.config, args.events, out_dir=None)
    fig = result.get("figure")
    if fig is not None:
        fig.show()
    network_fig = result.get("network_figure")
    if network_fig is not None:
        network_fig.show()


if __name__ == "__main__":
    main()
