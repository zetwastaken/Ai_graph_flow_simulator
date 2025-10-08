"""Interface layer for the monitoring project.

This module provides both a traditional CLI as well as a basic GUI that allows
users to run the simulator by selecting the required files through dialogs.
"""

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

# Import simulate with fallback to absolute import when executed as a script
try:
    # When run via ``python -m project.cli`` the package context exists
    from .sim.generator import plot_network_overview, plot_timeseries, simulate  # type: ignore
except Exception:
    # When run as a script the relative import will fail; fall back to local package
    from sim.generator import plot_network_overview, plot_timeseries, simulate  # type: ignore

try:
    from .viz.interactive_network import build_interactive_network_figure  # type: ignore
except Exception:  # pragma: no cover - fallback for script execution
    try:
        from viz.interactive_network import build_interactive_network_figure  # type: ignore
    except Exception:  # pragma: no cover - optional dependency may be missing
        build_interactive_network_figure = None  # type: ignore[assignment]


def launch_gui() -> None:
    """Launch a minimal Tkinter GUI to run the simulator."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:  # pragma: no cover - GUI import failure
        raise RuntimeError("Tkinter is required for the GUI but could not be imported.") from exc

    base_dir = Path(__file__).resolve().parent
    default_topology = base_dir / "data" / "network_topology.json"
    default_config = base_dir / "config.yml"

    root = tk.Tk()
    root.title("Network Simulator")

    topology_var = tk.StringVar(value=str(default_topology) if default_topology.exists() else "")
    config_var = tk.StringVar(value=str(default_config) if default_config.exists() else "")
    events_var = tk.StringVar()
    out_var = tk.StringVar()
    status_var = tk.StringVar(value="Waiting for input")

    summary = tk.Text(root, height=6, width=60, state="disabled")
    show_timeseries_var = tk.BooleanVar(value=True)
    show_overview_var = tk.BooleanVar(value=True)
    show_interactive_var = tk.BooleanVar(value=False)

    def make_selector(label: str, text_var: "tk.StringVar", browse_callback) -> None:
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(frame, text=label, width=16, anchor="w").pack(side="left")
        entry = tk.Entry(frame, textvariable=text_var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(frame, text="Browse", command=browse_callback).pack(side="left")

    def browse_file(title: str, filetypes):
        return filedialog.askopenfilename(title=title, filetypes=filetypes)

    def browse_dir(title: str):
        return filedialog.askdirectory(title=title)

    def update_summary(text: str) -> None:
        summary.configure(state="normal")
        summary.delete("1.0", tk.END)
        summary.insert(tk.END, text)
        summary.configure(state="disabled")

    def run_simulation() -> None:
        topology_path = topology_var.get().strip()
        config_path = config_var.get().strip()
        events_path = events_var.get().strip() or None
        out_dir = out_var.get().strip() or None

        if not topology_path or not os.path.isfile(topology_path):
            messagebox.showerror("Missing topology", "Please select a valid network topology JSON file.")
            return
        if not config_path or not os.path.isfile(config_path):
            messagebox.showerror("Missing config", "Please select a valid configuration YAML file.")
            return
        if events_path and not os.path.isfile(events_path):
            messagebox.showerror("Invalid events file", "The selected events file does not exist.")
            return
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Output directory error", f"Could not create output directory: {exc}")
                return

        status_var.set("Running simulation...")
        update_summary("")

        def worker() -> None:
            try:
                result = simulate(topology_path, config_path, events_path, out_dir=out_dir, make_plot=False)
            except Exception as err:
                root.after(0, lambda: messagebox.showerror("Simulation error", str(err)))
                root.after(0, lambda: status_var.set("Simulation failed."))
                return

            def on_complete() -> None:
                meters_df = result["meters_df"]
                edges_df = result["edges_df"]
                labels_df = result["labels_df"]
                lines = [
                    f"Simulated {len(meters_df)} meter readings.",
                    f"Simulated {len(edges_df)} edge records.",
                ]
                if labels_df is not None and not labels_df.empty:
                    lines.append(f"Injected {len(labels_df)} anomalies.")
                else:
                    lines.append("No anomalies injected.")
                if out_dir:
                    lines.append(f"Outputs saved to: {out_dir}")
                update_summary("\n".join(lines))
                status_var.set("Simulation complete.")
                show_timeseries = show_timeseries_var.get()
                show_overview = show_overview_var.get()
                show_interactive = show_interactive_var.get()
                try:
                    ts_fig = None
                    network_fig = None
                    graph_obj: Any = result.get("graph")
                    if show_timeseries or out_dir:
                        ts_fig = plot_timeseries(meters_df, labels_df)
                    if (show_overview or out_dir) and graph_obj is not None:
                        network_fig = plot_network_overview(graph_obj, meters_df, edges_df, labels_df)
                    if ts_fig is not None and show_timeseries:
                        ts_fig.show()
                    if network_fig is not None and show_overview:
                        network_fig.show()
                    if out_dir:
                        if ts_fig is not None:
                            plot_path = os.path.join(out_dir, "timeseries.svg")
                            ts_fig.savefig(plot_path, format="svg")
                        if network_fig is not None:
                            network_path = os.path.join(out_dir, "network_overview.svg")
                            network_fig.savefig(network_path, format="svg")
                except Exception as plot_err:
                    messagebox.showwarning("Plot error", f"Could not render plot: {plot_err}")

                if show_interactive:
                    graph_obj = result.get("graph")
                    if not callable(build_interactive_network_figure):
                        messagebox.showwarning(
                            "Interactive viewer unavailable",
                            "Plotly-based interactive viewer is not available in this environment.",
                        )
                    elif graph_obj is None:
                        messagebox.showwarning(
                            "Interactive viewer error",
                            "Simulation did not return graph data required for the interactive viewer.",
                        )
                    else:
                        try:
                            interactive_fig = build_interactive_network_figure(graph_obj, edges_df, meters_df)  # type: ignore[misc]
                            interactive_fig.show()
                        except Exception as interactive_err:
                            messagebox.showwarning(
                                "Interactive viewer error",
                                f"Could not render interactive viewer: {interactive_err}",
                            )

            root.after(0, on_complete)

        threading.Thread(target=worker, daemon=True).start()

    make_selector("Topology JSON:", topology_var, lambda: topology_var.set(browse_file("Select topology JSON", [("JSON files", "*.json"), ("All files", "*")]) or topology_var.get()))
    make_selector("Config YAML:", config_var, lambda: config_var.set(browse_file("Select configuration", [("YAML files", "*.yaml *.yml"), ("All files", "*")]) or config_var.get()))
    make_selector("Events (optional):", events_var, lambda: events_var.set(browse_file("Select events file", [("CSV files", "*.csv"), ("YAML files", "*.yaml *.yml"), ("All files", "*")]) or events_var.get()))
    make_selector("Output directory:", out_var, lambda: out_var.set(browse_dir("Select output directory") or out_var.get()))

    viz_frame = tk.LabelFrame(root, text="Visualisations")
    viz_frame.pack(fill="x", padx=10, pady=(5, 5))
    tk.Checkbutton(viz_frame, text="Timeseries plot", variable=show_timeseries_var).pack(anchor="w")
    tk.Checkbutton(viz_frame, text="Network overview", variable=show_overview_var).pack(anchor="w")
    interactive_available = callable(build_interactive_network_figure)
    if not interactive_available:
        show_interactive_var.set(False)
    tk.Checkbutton(
        viz_frame,
        text="Interactive network (requires Plotly)",
        variable=show_interactive_var,
        state="normal" if interactive_available else "disabled",
    ).pack(anchor="w")

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=10, pady=10)
    tk.Button(button_frame, text="Run simulation", command=run_simulation).pack(side="left")

    status_frame = tk.Frame(root)
    status_frame.pack(fill="x", padx=10)
    tk.Label(status_frame, textvariable=status_var, anchor="w").pack(fill="x")

    summary_frame = tk.Frame(root)
    summary_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
    tk.Label(summary_frame, text="Summary:", anchor="w").pack(fill="x")
    summary.pack(fill="both", expand=True)

    root.mainloop()


def main() -> None:
    if len(sys.argv) == 1:
        launch_gui()
        return

    parser = argparse.ArgumentParser(description="Network monitoring CLI")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of the CLI")
    subparsers = parser.add_subparsers(dest="command")

    # simulate subcommand
    sim_parser = subparsers.add_parser("simulate", help="Run the data simulator")
    sim_parser.add_argument("--topology", required=True, help="Path to network_topology.json")
    sim_parser.add_argument("--config", required=True, help="Path to configuration YAML")
    sim_parser.add_argument("--events", default=None, help="Path to events CSV/YAML file")
    sim_parser.add_argument(
        "--out", default=None, help="Output directory for CSVs and plot; if omitted no files are saved"
    )

    args = parser.parse_args()

    if args.gui or args.command is None:
        launch_gui()
        return

    if args.command == "simulate":
        out_dir = args.out
        result = simulate(args.topology, args.config, args.events, out_dir=out_dir)
        # Print summary to stdout
        meters_df = result["meters_df"]
        edges_df = result["edges_df"]
        labels_df = result["labels_df"]
        print(f"Simulated {len(meters_df)} meter readings and {len(edges_df)} edge records.")
        if labels_df is not None and not labels_df.empty:
            print(f"Injected {len(labels_df)} anomalies.")
        else:
            print("No anomalies injected.")
        fig = result.get("figure")
        network_fig = result.get("network_figure")
        if fig is not None:
            fig.show()
        if network_fig is not None:
            network_fig.show()


if __name__ == "__main__":
    main()
