# Filename : InfluenceMaxPolicy.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import time

import matplotlib.pyplot as plt
import numpy as np

from MonteCarloSimulation import *
from Policies.SocialCELFppPolicy import *
from Policies.SocialDegreeDiscountIC import *
from Policies.SocialDegreeDiscountPolicyV2 import *
from Policies.SocialGreedyPolicy import *
from Policies.SocialRandomPolicy import *
from Policies.SocialSingleDegreeDiscountPolicy import *


class SimplePolicies:

    def __init__(self, network, mc_iterations, bandit='false', max_iters=None):
        self.network = network
        self.mc_iterations = mc_iterations
        self.influence_spread = {}
        self.opt_seeds = {}
        self.bandit = bandit
        self.max_iters = network.number_of_nodes() \
            if max_iters is None \
            else max_iters
        self.policies = {'grd': SocialGreedyPolicy, 'rnd': SocialRandomPolicy,
                         'celfpp': SocialCELFppPolicy, 'sdd': SocialSingleDegreeDiscountPolicy,
                         'ddic': SocialDegreeDiscountICPolicy, 'ddicv2': SocialDegreeDiscountICPolicyV2}

    def execute(self, budget, policies):

        for policy_type in policies:
            start = time.time()
            influence_spread = [0]
            opt_seeds = {}

            mc = MonteCarloSimulation(self.network, self.mc_iterations)
            policy = self.policies[policy_type](self.network.copy(), budget, mc)


            while sum(opt_seeds.values()) < budget \
                    and len(opt_seeds) < self.network.number_of_nodes():
                seeds, infl_spread = \
                    policy.run_policy(self.network, budget, mc)
                influence_spread.append(infl_spread)
                opt_seeds.update(seeds)

            self.influence_spread[policy_type] = influence_spread
            self.opt_seeds[policy_type] = opt_seeds
            end = time.time() - start
            if not self.bandit:
                print("algorithm " + policy_type + " running time: " + str(end))

    def plot_results(self):
        for k, opt_seeds in self.opt_seeds.items():
            print(k + ' optimal set of seeds: ', end='')
            print(opt_seeds)

        for inf_spread in self.influence_spread.values():
            print('influence spread: ' + str(inf_spread))
            l = len(inf_spread)
            if l < self.network.number_of_nodes() < 50:
                for _ in range(l, self.network.number_of_nodes() + 1):
                    inf_spread.append(inf_spread[l - 1])
            plt.plot(range(len(inf_spread)), inf_spread)

        max_y = max([np.max(i) for i in self.influence_spread.values()])
        plt.axis([0, max([len(i) for i in self.influence_spread.values()]),
                  -.001, max_y * 1.05 if max_y > .05 else .10])

        plt.legend(self.influence_spread.keys())
        plt.xlabel('Number of Seeds')
        plt.ylabel('Influence spread')
