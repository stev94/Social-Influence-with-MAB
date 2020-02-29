# Filename : Main.py
# Date : 2019/11/12 22.58
# Project: Social Influence
# Author : Stefano Valladares

import json
import random

import networkx as nx
import numpy as np


def create_network(n_nodes, n_features, max_degree=None):
    if max_degree is not None:
        network = generate_graph_max_neighbors(n_nodes, max_degree)
    else:
        network = nx.gnp_random_graph(n_nodes, np.log(n_nodes) / n_nodes,
                                      directed=True)
    for node in network.nodes.values():
        node['cost'] = 1

    theta = np.random.dirichlet(np.ones(n_features), size=1)
    for edge in network.edges.values():
        edge['features'] = np.random.binomial(1, 0.25, size=n_features).tolist()
        edge['prob'] = np.dot(theta, edge['features'])[0]
        if edge['prob'] > 1: edge['prob'] = 1

    return network


def create_celf_network(n_nodes, n_features, max_degree=None):
    if max_degree is not None:
        network = generate_graph_max_neighbors(n_nodes, max_degree)
    else:
        network = nx.gnp_random_graph(n_nodes, np.log(n_nodes) / n_nodes,
                                      directed=True)
    for node in network.nodes.values():
        node['cost'] = 1
        node['mg1'] = 0
        node['prev_best'] = None
        node['mg2'] = 0
        node['flag'] = None

    theta = np.random.dirichlet(np.ones(n_features), size=1)
    for edge in network.edges.values():
        edge['features'] = np.random.binomial(1, 0.25, size=n_features)
        edge['prob'] = np.dot(theta, edge['features'])
        edge['features'] = np.random.binomial(1, 0.5, size=n_features).tolist()
        edge['prob'] = np.dot(theta, edge['features'])[0]
        if edge['prob'] > 1: edge['prob'] = 1

    return network


def generate_graph_max_neighbors(n_nodes, max_degree):
    degrees = []
    for _ in range(n_nodes):
        degrees.append(random.randint(0, max_degree))

    network = nx.directed_configuration_model(degrees,
                                              np.random.permutation(degrees))
    network = nx.DiGraph(network)
    network.remove_edges_from(nx.selfloop_edges(network))

    return network


def store_network(network, filename):
    data = nx.readwrite.node_link_data(network)
    data_folder = Path('../Documentation/Graphs/')

    with open(data_folder / str(filename + '.txt'), 'w') as outfile:
        json.dump(data, outfile)


def load_network(filename):
    data_folder = Path('../Documentation/Graphs/')

    with open(data_folder / str(filename + '.txt')) as jsonfile:
        data = json.load(jsonfile)

    return nx.readwrite.node_link_graph(data)
