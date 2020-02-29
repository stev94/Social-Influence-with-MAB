# Filename : MainGreedy.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

from SocialMain import *

n_nodes = [150]

main = SocialMain(n_features=4,
                  n_nodes=n_nodes,
                  mc_iterations=[3000],
                  budgets=[int(n_nodes * 0.15) for n_nodes in n_nodes],
                  alg_type='known')

main.run_algs(policies=['celfpp', 'ddic', 'sdd', 'rnd'])
