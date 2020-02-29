# Filename : CUCB.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

from .SocialPolicy import *


class CUCB(SocialPolicy):

    def __init__(self, network, mc_iterations):
        super().__init__(network, mc_iterations)
        self.estimates = {}
        self.t = 0
        for edge in network.edges:
            self.estimates[edge] = np.zeros(2, float)

    def compute_probs(self):
        if self.t == 0:
            for arm in self.estimates.keys():
                self.network.edges[arm]['prob'] = 1
            return
        for arm, params in self.estimates.items():
            self.network.edges[arm]['prob'] = \
                params[0] + np.sqrt(3 * np.log(self.t) / (2 * params[1])) \
                if params[1] > 0 else 1
        self.normalize_probs()

    def normalize_probs(self):
        max_prob = np.max([p[2]
                           for p in list(self.network.edges.data('prob'))])
        max_prob = 1 if max_prob <= 0 else max_prob
        for edge in self.network.edges.values():
            edge['prob'] = edge['prob'] / max_prob

    def update_estimation(self, pulled_arms, failed_pulled_arms):
        self.t += 1
        for arm in self.network.edges:
            if arm in pulled_arms:
                self.estimates[arm][1] += 1
                l = self.estimates[arm][1]
                self.estimates[arm][0] = (self.estimates[arm][0] * (l - 1) + 1) / l
            elif arm in failed_pulled_arms:
                self.estimates[arm][1] += 1
                l = self.estimates[arm][1]
                self.estimates[arm][0] = (self.estimates[arm][0] * (l - 1)) / l
