"""Tests for network loading and routing logic."""

from project.sim.generator import build_graph, assign_routes


def test_build_graph_and_assign_routes():
    """Verify that the sample topology loads correctly and that routes are assigned."""
    # Load the sample topology in the data folder
    topology_path = "project/data/network_topology.json"
    G = build_graph(topology_path)
    # Verify node count and types
    assert len(G.nodes) == 4, "Should load four nodes from sample topology"
    assert len(G.edges) == 3, "Should load three edges from sample topology"
    # All sources and consumers must have meter_id
    for node in G.nodes.values():
        if node.type in ("source", "consumer"):
            assert node.meter_id is not None, f"Node {node.id} should have a meter_id"
    # Assign routes and ensure each consumer maps to the sole source
    routes, edge_to_consumers, edge_to_sources = assign_routes(G)
    # There should be two consumers
    assert len(routes) == 2, "Two consumers should be assigned routes"
    for consumer_id, route in routes.items():
        assert route["source"] == "source_1", f"Consumer {consumer_id} should be supplied by source_1"
        # Path should include exactly the correct edge sequence
        if consumer_id == "consumer_1":
            assert route["path_edges"] == ["e1", "e2"]
        elif consumer_id == "consumer_2":
            assert route["path_edges"] == ["e1", "e3"]