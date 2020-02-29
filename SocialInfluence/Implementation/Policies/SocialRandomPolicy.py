# Filename : SocialRandomPolicy.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import random

import numpy as np


class SocialRandomPolicy:

    def __init__(self, network, budget, mc):
        self.opt_seeds = {}

    def run_policy(self, network, budget, mc):
        influence_spread = 0

        rnd_node = self.select_rnd(network, self.opt_seeds.values())
        act_probs = mc.run(list(self.opt_seeds.keys()) + [rnd_node])
        self.opt_seeds[rnd_node] = network.nodes[rnd_node]['cost']
        influence_spread = np.sum(act_probs) / network.nodes[rnd_node]['cost']

        return self.opt_seeds, influence_spread

    def select_rnd(self, network, opt_seeds):
        while True:
            rnd_node = random.randint(0, network.number_of_nodes() - 1)
            if rnd_node not in opt_seeds:
                return rnd_node
