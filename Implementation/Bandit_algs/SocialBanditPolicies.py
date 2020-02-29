# Filename : SocialBandit.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import matplotlib.pyplot as plt

from .CTS import *
from .CUCB import *
from .LinearUCB import *


class SocialBanditPolicies:

    def __init__(self, network, budget, mc_iterations, env):
        self.network = network
        self.env = env
        self.budget = budget
        self.mc_iterations = mc_iterations
        self.rew_per_exp = {}
        self.policies = {'lin_ucb': LinearUCB, 'cucb': CUCB, 'cts': CTS}

    def execute(self, n_exp, th, policies):
        for n in range(n_exp):
            print('Experiment: ' + str(n))
            for policy_type in policies:
                self.rew_per_exp[policy_type] = []
                policy = self.policies[policy_type](self.network.copy(),
                                                    self.mc_iterations)
                for t in range(th):
                    seeds, _ = policy.select_arms(self.budget)
                    act_edges, failed_act_edges, infl_spread = self.env.round(list(seeds.keys()))
                    print('time horizon: ' + str(t) + ' act nodes  ' + str(seeds) + ' act edges ' + str(
                        act_edges) + ' influence spread: ' + str(infl_spread))
                    policy.update(seeds, act_edges, failed_act_edges, infl_spread)

                self.rew_per_exp[policy_type].append(policy.collected_reward)

    def plot_results(self):
        opt = self.env.opt(self.mc_iterations, self.budget)

        plt.title(' n_nodes =' + str(self.network.number_of_nodes()) +
                  ' budget =' + str(self.budget) +
                  ' mc_iter =' + str(self.mc_iterations))

        plt.ylabel("Regret")
        plt.xlabel("t")
        for rew in self.rew_per_exp.values():
            plt.plot(np.cumsum(np.mean(opt[-1] - rew, axis=0)))
        plt.legend([k for k in self.rew_per_exp.keys()])
