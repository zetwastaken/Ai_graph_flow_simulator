"""Project-level CLI proxy to the high-level simulator."""

from __future__ import annotations

from datetime import datetime

from . import FlowSimulator, SimulationConfig


def run():
    """
    Run a default simulation with basic configuration.

    Creates and executes a complete simulation pipeline including setup,
    data generation, saving, visualization, and reporting.
    """
    config = SimulationConfig(
        start_time=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    )
    simulator = FlowSimulator(config)
    simulator.setup()
    simulator.run()
    simulator.save_data()
    simulator.visualize()
    simulator.print_summary()


if __name__ == "__main__":
    run()
