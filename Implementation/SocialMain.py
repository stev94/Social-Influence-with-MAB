# Filename : SocialMain.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import matplotlib.pyplot as plt

import GraphUtils as ut
from Bandit_algs.SocialBanditPolicies import *
from Bandit_algs.SocialEnvironment import *


class SocialMain:

    def __init__(self, n_features, n_nodes, mc_iterations, budgets, alg_type):
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.mc_iterations = mc_iterations
        self.budgets = budgets
        self.alg_type = alg_type
        self.learners = None

    def run_algs(self, policies, n_experiments=None, time_horizon=None):
        for n_nodes, budget in zip(self.n_nodes, self.budgets):

            network = ut.create_network(n_nodes, self.n_features)
            print('\nBudget: ' + str(budget))
            print('Number of nodes: ' + str(n_nodes))

            env = None
            if self.alg_type == 'unknown':
                print('Time Horizon: ', str(time_horizon))
                print('Number of experiments: ', str(n_experiments))
                env = SocialEnvironment(network.copy())
                for e in network.edges.values():
                    e['prob'] = None

            for mc_iterations in self.mc_iterations:
                print('Montecarlo iterations: ' + str(mc_iterations))
                if self.alg_type == 'known':
                    self.learners = SimplePolicies(network, mc_iterations, bandit=False)
                    self.learners.execute(budget, policies)
                    self.plot_results()
                else:
                    self.learners = SocialBanditPolicies(network, budget,
                                                         mc_iterations, env)
                    self.learners.execute(n_experiments, time_horizon,
                                          policies)
                    self.plot_results()

    def plot_results(self):
        plt.figure(0)

        self.learners.plot_results()

        plt.show()
