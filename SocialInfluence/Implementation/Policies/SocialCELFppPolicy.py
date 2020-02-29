import numpy as np
from heapdict import heapdict


class SocialCELFppPolicy:

    def __init__(self, network, budget, mc):
        self.opt_seeds = {}
        self.network = network
        self.budget = budget
        self.Q = heapdict()
        self.last_seed = None
        self.cur_best = None
        self.node_data_list = []
        self.influence_spread = 0
        for node in self.network.nodes:
            self.network.nodes[node]['mg1'] = np.sum(mc.run([node]))
            self.network.nodes[node]['prev_best'] = self.cur_best
            self.network.nodes[node]['mg2'] = np.sum(mc.run([node, self.cur_best])) - self.network.nodes[self.cur_best][
                'mg1'] if self.cur_best else self.network.nodes[node]['mg1']
            self.network.nodes[node]['flag'] = 0
            self.cur_best = self.cur_best if self.cur_best and self.network.nodes[self.cur_best]['mg1'] > \
                                             self.network.nodes[node]['mg1'] else node
            self.Q[node] = -self.network.nodes[node]['mg1']

    def run_policy(self, network, budget, mc):
        while 1:
            node_idx, _ = self.Q.peekitem()
            if self.network.nodes[node_idx]['flag'] == len(list(self.opt_seeds.keys())):
                self.opt_seeds[node_idx] = self.network.nodes[node_idx]['cost']
                self.influence_spread += self.network.nodes[node_idx]['mg1']
                del self.Q[node_idx]
                self.last_seed = node_idx
                return self.opt_seeds, self.influence_spread
            elif self.network.nodes[node_idx]['prev_best'] == self.last_seed:
                self.network.nodes[node_idx]['mg1'] = self.network.nodes[node_idx]['mg2']
            else:
                before = np.sum(mc.run(list(self.opt_seeds.keys())))
                self.opt_seeds[node_idx] = self.network.nodes[node_idx]['cost']
                after = np.sum(mc.run(list(self.opt_seeds.keys())))
                del self.opt_seeds[node_idx]
                self.network.nodes[node_idx]['mg1'] = after - before
                self.network.nodes[node_idx]['prev_best'] = self.cur_best

                if self.cur_best not in list(self.opt_seeds.keys()):
                    self.opt_seeds[self.cur_best] = self.network.nodes[self.cur_best]['cost']
                    before = np.sum(mc.run(list(self.opt_seeds.keys())))
                    self.opt_seeds[node_idx] = self.network.nodes[node_idx]['cost']
                    after = np.sum(mc.run(list(self.opt_seeds.keys())))
                    del self.opt_seeds[self.cur_best]
                    if node_idx != self.cur_best: del self.opt_seeds[node_idx]
                    self.network.nodes[node_idx]['mg2'] = after - before
                else:
                    self.network.nodes[node_idx]['mg2'] = self.network.nodes[node_idx]['mg1']

            if self.cur_best and self.network.nodes[self.cur_best]['mg1'] < self.network.nodes[node_idx]['mg1']:
                self.cur_best = node_idx

            self.network.nodes[node_idx]['flag'] = len(list(self.opt_seeds.keys()))
            self.Q[node_idx] = - self.network.nodes[node_idx]['mg1']
