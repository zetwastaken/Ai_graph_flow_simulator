#!/usr/bin/env python3
"""Unified launcher for the AI Graph Flow Simulator stack."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import time

# Optional dependencies are imported lazily
try:  # pragma: no cover - optional import
    import uvicorn
except ImportError:  # pragma: no cover - handled at runtime
    uvicorn = None  # type: ignore

try:
    from project import FlowSimulator, SimulationConfig, StreamingSimulator
    from project.api.server import app as api_app, configure_runtime
    from project.auth.security import JWTAuthenticator
except ModuleNotFoundError as exc:  # pragma: no cover - runtime check
    missing = exc.name or "dependency"
    print(f"[ERROR] Missing dependency '{missing}'. Run 'pip install -r requirements.txt' and retry.")
    sys.exit(1)


def _start_thread(target, name: str) -> threading.Thread:
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread


def main():
    parser = argparse.ArgumentParser(description="Run simulation, API and dashboard in one command.")
    parser.add_argument("--nodes", type=int, default=20, help="Number of consumer nodes")
    parser.add_argument("--sources", type=int, default=1, help="Number of supply sources")
    parser.add_argument("--duration", type=int, default=24, help="Simulation duration in hours")
    parser.add_argument("--sampling", type=float, default=0.1, help="Sampling frequency in Hz")
    parser.add_argument("--topology", choices=["tree", "mesh", "random"], default="tree",
                        help="Topology type")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--export", choices=["csv", "json"], default="csv", help="Export format")
    parser.add_argument("--real-time", action="store_true",
                        help="Enable real-time streaming delays (sleep between samples)")
    parser.add_argument("--no-api", action="store_true", help="Skip launching the FastAPI server")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching the Dash dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Port for the dashboard server")
    args = parser.parse_args()

    config = SimulationConfig(
        num_nodes=args.nodes,
        num_sources=args.sources,
        duration_hours=args.duration,
        sampling_frequency_hz=args.sampling,
        topology_type=args.topology,
        output_dir=args.output,
        export_format=args.export,
        real_time_mode=args.real_time,
    )

    print("Running baseline simulation...")
    simulator = FlowSimulator(config)
    simulator.setup()
    simulator.run()
    simulator.save_data()
    simulator.visualize()
    simulator.print_summary()

    print("\nStarting streaming engine...")
    streamer = StreamingSimulator(simulator)
    streamer.start_background(real_time=config.real_time_mode)

    # Configure API runtime to reuse this simulator
    auth = JWTAuthenticator(config.auth_public_key, config.jwt_audience)
    configure_runtime(simulator, streamer, auth)

    threads = []

    if not args.no_api and uvicorn is None:
        print("[WARN] uvicorn not installed – skipping API server. Run 'pip install uvicorn' to enable it.")
        args.no_api = True

    if not args.no_api:
        def run_api():
            uvicorn.run(
                api_app,
                host=config.websocket_host,
                port=config.websocket_port,
                log_level="info",
            )

        threads.append(_start_thread(run_api, "api-server"))
        snapshot_url = f"http://{config.websocket_host}:{config.websocket_port}/snapshot"
    else:
        snapshot_url = f"file://{os.path.abspath(config.output_dir)}/flow_measurements.csv"
    os.environ.setdefault("SIMULATOR_SNAPSHOT_URL", snapshot_url)

    os.environ.setdefault("SIMULATOR_OUTPUT", config.output_dir)

    if not args.no_dashboard:
        # Lazy import after environment variables are configured
        try:
            dashboard_module = importlib.import_module("project.dashboard.app")
        except ImportError:
            print("[WARN] Dash dashboard dependencies missing – skipping dashboard.")
            args.no_dashboard = True
        else:
            dash_app = dashboard_module.create_app()

            def run_dash():
                dash_app.run_server(host="0.0.0.0", port=args.dashboard_port, debug=False)

            threads.append(_start_thread(run_dash, "dashboard"))
            print(f"Dashboard available on http://localhost:{args.dashboard_port}")

    if not threads:
        print("All tasks completed.")
        streamer.stop()
        return

    print("\nServices running. Press Ctrl+C to stop.")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
