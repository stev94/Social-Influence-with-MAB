import numpy as np


class SocialDegreeDiscountICPolicy:

    def __init__(self, network, budget, mc):
        self.opt_seeds = {}

        self.ddv = np.zeros(len(list(network.nodes())))
        self.tv = np.zeros(len(list(network.nodes())))
        for i, j in list(network.degree):
            self.ddv[i] = j

    #        self.ddv = np.zeros(len(list(network.nodes())))
    #        for i in range(graph.node_num):
    #            ddv[i] = graph.get_out_degree(i + 1)

    def run_policy(self, network, budget, mc):
        node = self.ddv.argmax()
        self.opt_seeds[node] = network.nodes[node]['cost']
        self.ddv[node] = -1
        children = network.successors(node)
        for child in children:
            if child not in list(self.opt_seeds.keys()):
                self.tv[child] += 1
                self.ddv[child] += -2 * self.tv[child] - \
                                   (network.degree(child) - self.tv[child]) * \
                                   self.tv[child] * network[node][child]['prob']

        return self.opt_seeds, sum(mc.run(list(self.opt_seeds.keys())))
