# Filename : SocialLearner.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

from abc import ABC, abstractmethod

from Policies.SimplePolicies import *


class SocialPolicy(ABC):

    def __init__(self, network, mc_iterations, oracle_policy='celfpp'):
        self.network = network
        self.mc_iterations = mc_iterations
        self.collected_reward = []
        self.oracle_policy = oracle_policy

    def select_arms(self, budget):
        self.compute_probs()
        oracle = SimplePolicies(self.network, self.mc_iterations, budget)

        oracle.execute(budget, [self.oracle_policy])

        return oracle.opt_seeds[self.oracle_policy], \
               oracle.influence_spread[self.oracle_policy]

    def update(self, opt_seeds, act_edges, failed_act_edges, reward):
        pulled_arms = []
        failed_pulled_arms = []
        for edge in self.network.edges:
            if edge in act_edges:
                pulled_arms.append(edge)
            elif edge in failed_act_edges:
                failed_pulled_arms.append(edge)

        self.collected_reward.append(reward)
        self.update_estimation(act_edges, failed_act_edges)

    @abstractmethod
    def compute_probs(self):
        pass

    @abstractmethod
    def update_estimation(self, pulled_arms, failed_pulled_arms):
        pass
