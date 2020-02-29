# Filename : LinearUCB.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import numpy as np

from .CUCB import *


class LinearUCB(CUCB):

    def __init__(self, network, mc_iterations):
        super().__init__(network, mc_iterations)
        self.c = 2.0
        self.X = {}
        for edge, atrs in network.edges.items():
            self.X[edge] = atrs['features']
        self.M = np.identity(len(list(self.X.values())[0]))
        self.b = np.zeros((len(list(self.X.values())[0]), 1))

    def compute_probs(self):
        theta = np.dot(self.M, self.b)
        for arm, features in self.X.items():
            features = np.atleast_2d(features).T
            self.network.edges[arm]['prob'] = (
                    np.dot(theta.T, features) + self.c
                    * np.sqrt(np.dot(features.T,
                                     np.dot(self.M, features)))[0][0])
        self.normalize_probs()

    def update_estimation(self, pulled_arms, failed_pulled_arms):
        for arm in self.network.edges:
            if arm in pulled_arms:
                a = np.atleast_2d(self.X[arm]).T
                self.M -= np.dot(self.M, np.dot(a, np.dot(a.T, self.M))) \
                          / (np.dot(a.T, np.dot(self.M, a)) + 1)
                self.b += a
            elif arm in failed_pulled_arms:
                a = np.atleast_2d(self.X[arm]).T
                self.M -= np.dot(self.M, np.dot(a, np.dot(a.T, self.M))) \
                          / (np.dot(a.T, np.dot(self.M, a)) + 1)
