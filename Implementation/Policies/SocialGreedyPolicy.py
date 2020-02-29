# Filename : SocialGreedyPolicy.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import numpy as np


class SocialGreedyPolicy:

    def __init__(self, network, budget, mc):
        self.opt_seeds = {}

    def run_policy(self, network, budget, mc):
        marg_incr = {}
        influence_spread = 0

        for node in set(network.nodes) - set(self.opt_seeds.keys()):
            act_probs = mc.run(list(self.opt_seeds.keys()) + [node])
            marg_incr[node] = np.sum(act_probs) \
                              / network.nodes[node]['cost']
        rank = sorted(marg_incr, key=marg_incr.get, reverse=True)

        for i in range(len(rank)):
            cost = network.nodes[rank[i]]['cost']
            if sum(self.opt_seeds.values()) + cost <= budget:
                self.opt_seeds[rank[i]] = cost
                influence_spread = marg_incr[rank[i]]
                break
            elif i == len(rank) - 1:
                influence_spread = influence_spread[-1]

        return self.opt_seeds, influence_spread
