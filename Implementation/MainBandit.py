# Filename : MainLinUCB.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

from SocialMain import *

n_nodes = np.array([20])

main = SocialMain(n_features=4,
                  n_nodes=n_nodes,
                  mc_iterations=[500],
                  budgets=[int(n_nodes * .2) for n_nodes in n_nodes],
                  alg_type='unknown')

main.run_algs(policies=['lin_ucb', 'cucb', 'cts'],
              n_experiments=50,
              time_horizon=200)
