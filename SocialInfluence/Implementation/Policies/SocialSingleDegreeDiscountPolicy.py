class SocialSingleDegreeDiscountPolicy:

    def __init__(self, network, budget, mc):
        self.opt_seeds = {}
        self.degree_count = dict(network.degree)
        self.neighborhood_fn = network.predecessors

    def run_policy(self, network, budget, mc):
        node = max(self.degree_count.items(), key=lambda x: x[1])[0]
        self.opt_seeds[node] = network.nodes[node]['cost']
        for neighbor in self.neighborhood_fn(node):
            self.degree_count[neighbor] -= 1
        self.degree_count[node] = 0

        return self.opt_seeds, sum(mc.run(list(self.opt_seeds.keys())))
