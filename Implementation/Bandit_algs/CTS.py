# Filename : CTS.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares


from .SocialPolicy import *


class CTS(SocialPolicy):

    def __init__(self, network, mc_iterations):
        super().__init__(network, mc_iterations)
        self.beta_parameters = {}
        for edge in network.edges:
            self.beta_parameters[edge] = np.ones(2, float)

    def compute_probs(self):
        for arm, params in self.beta_parameters.items():
            self.network.edges[arm]['prob'] = np.random.beta(params[0],
                                                             params[1])

    def update_estimation(self, pulled_arms, failed_pulled_arms):
        for arm in self.network.edges:
            if arm in pulled_arms:
                self.beta_parameters[arm][0] += 1
            elif arm in failed_pulled_arms:
                self.beta_parameters[arm][1] += 1
