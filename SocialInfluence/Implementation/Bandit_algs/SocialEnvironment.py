# Filename : ScialEnvironment.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import numpy as np

from Policies.SimplePolicies import *


class SocialEnvironment:

    def __init__(self, network):
        self.network = network

    def round(self, act_nodes):
        network = self.network.copy()
        return self.simulate_cascade(act_nodes, [], [], len(act_nodes), network)

    def simulate_cascade(self, act_nodes, act_edges, failed_act_edges, infl_spread, network):
        influenced_nodes = []

        if not act_nodes:
            return act_edges, failed_act_edges, infl_spread

        for node in list(act_nodes):
            for edge in network.edges(node):
                if edge[1] not in act_nodes + influenced_nodes:
                    if np.random.binomial(1, network.edges[edge]['prob']):
                        influenced_nodes.append(edge[1])
                        infl_spread += 1
                        act_edges.append(edge)
                    else:
                        failed_act_edges.append(edge)
            network.remove_node(node)

        return self.simulate_cascade(influenced_nodes, act_edges, failed_act_edges,
                                     infl_spread, network)

    def opt(self, mc_iterations, budget):
        oracle = SimplePolicies(self.network, 3000, max_iters=budget, bandit=True)
        oracle.execute(budget, ['celfpp'])

        print(' act edges  ' + str(oracle.opt_seeds) + ' influence spread:  ' + str(oracle.influence_spread['celfpp']))

        return oracle.influence_spread['celfpp']
