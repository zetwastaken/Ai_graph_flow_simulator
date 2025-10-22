# -*- coding: utf-8 -*-
"""
Main script to run the network flow simulation.
"""
import random
import time
from config import SIMULATION_CONFIG, ANOMALY_TYPES, OUTPUT_DATA_CSV, OUTPUT_REPORT_TXT
from topology.graph_generator import GraphGenerator
from simulation.flow_simulator import FlowSimulator
from anomalies.anomaly_injector import AnomalyInjector
from utils.data_saver import DataSaver
from visualization.plotter import Plotter

def main():
    """
    Main function to execute the simulation.
    """
    print("Starting simulation...")
    start_time = time.time()

    # 1. Generate topology
    graph_gen = GraphGenerator(SIMULATION_CONFIG['num_nodes'], SIMULATION_CONFIG['num_sources'])
    network_graph = graph_gen.generate_topology()

    # 2. Initialize simulation components
    flow_sim = FlowSimulator(network_graph, SIMULATION_CONFIG)
    anomaly_injector = AnomalyInjector(ANOMALY_TYPES)
    
    # 3. Run simulation
    all_data = []
    duration = SIMULATION_CONFIG['simulation_duration_seconds']
    freq = SIMULATION_CONFIG['sampling_frequency_hz']
    
    for t in range(0, duration, 1//freq):
        # Introduce anomalies randomly
        if random.random() < SIMULATION_CONFIG['anomaly_frequency']:
            anomaly_injector.introduce_anomaly(network_graph, t)
            
        # Simulate one time step
        step_data = flow_sim.simulate_step(t, anomaly_injector)
        all_data.extend(step_data)

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

    # 4. Save data
    saver = DataSaver(OUTPUT_DATA_CSV)
    saver.save_to_csv(all_data)

    # 5. Generate and save report
    num_anomalies = len(anomaly_injector.active_anomalies)
    report = f"Simulation Report\n\n"
    report += f"Total duration: {duration} seconds\n"
    report += f"Number of nodes: {SIMULATION_CONFIG['num_nodes']}\n"
    report += f"Number of data points: {len(all_data)}\n"
    report += f"Number of anomalies injected: {num_anomalies}\n"
    saver.save_report(report, OUTPUT_REPORT_TXT)

    # 6. Visualize results
    plotter = Plotter(OUTPUT_DATA_CSV)
    plotter.load_data()
    if network_graph and SIMULATION_CONFIG['num_nodes'] > 0:
        # Plot a few random nodes
        random_nodes = random.sample(list(network_graph.nodes()), min(3, SIMULATION_CONFIG['num_nodes']))
        for node_id in random_nodes:
            plotter.plot_flow_for_node(node_id)

if __name__ == "__main__":
    main()
