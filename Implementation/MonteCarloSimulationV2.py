# Filename : Main.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import networkx as nx
import numpy as np


class MonteCarloSimulationV2:

    def __init__(self, network, iterations):
        self.network = network
        self.iterations = iterations

    def run(self, seeds):
        act_prob = np.zeros(self.network.number_of_nodes())
        nodes = set(seeds)
        for seed in [seed for seed in seeds if self.network.has_node(seed)]:
            nodes.update(set(nx.dfs_preorder_nodes(self.network, seed)))
        network_with_no_isolated_nodes = self.network.subgraph(nodes)

        for _ in range(self.iterations):
            act_prob = self.run_simulation(seeds, act_prob, network_with_no_isolated_nodes)
        return act_prob / self.iterations

    def run_simulation(self, seeds, act_prob, network_with_no_isolated_nodes):
        live_graph = self.generate_live_edge_graph(seeds, network_with_no_isolated_nodes)
        nodes = set(seeds)
        for seed in [seed for seed in seeds if live_graph.has_node(seed)]:
            nodes.update(set(nx.dfs_preorder_nodes(live_graph, seed)))

        return act_prob

    def generate_live_edge_graph(self, seeds, network_with_no_isolated_nodes):
        live_edges = []
        for edge in set(network_with_no_isolated_nodes.edges):
            if np.random.binomial(1, network_with_no_isolated_nodes.edges[edge]['prob']): live_edges.append(edge)
        return network_with_no_isolated_nodes.edge_subgraph(live_edges)
