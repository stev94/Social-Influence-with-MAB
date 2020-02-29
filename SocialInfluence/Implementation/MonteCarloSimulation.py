# Filename : MonteCarloSimulation.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import networkx as nx
import numpy as np


class MonteCarloSimulation:

    def __init__(self, network, iterations):
        self.network = network
        self.iterations = iterations

    def run(self, seeds):
        act_prob = np.zeros(self.network.number_of_nodes())
        for _ in range(self.iterations):
            #            start_time = time.time()
            act_prob = self.run_simulation(seeds, act_prob)
            #            print("--- %s seconds ---" % (time.time() - start_time))
        return act_prob / self.iterations

    def run_simulation(self, seeds, act_prob):
        live_graph = self.generate_live_edge_graph()
        nodes = set(seeds)

        for seed in [seed for seed in seeds if live_graph.has_node(seed)]:
            nodes.update(set(nx.dfs_preorder_nodes(live_graph, seed)))
        act_prob[list(nodes)] += 1

        return act_prob

    def generate_live_edge_graph(self):
        live_edges = []

        for edge in set(self.network.edges):
            if np.random.binomial(1, self.network.edges[edge]['prob']):
                live_edges.append(edge)

        return self.network.edge_subgraph(live_edges)
