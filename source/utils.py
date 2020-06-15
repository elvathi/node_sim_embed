import numpy as np
import random
import networkx as nx

def get_graph(filename, directed):
    ''' Takes the name of the dataset,
        reads the edgelist
        and returns the corresponding networkx graph '''
    if directed:
        G = nx.read_edgelist(filename, create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(filename)
    return G

def get_neighbours(G):
    ''' Takes a networkx graph and returns a dictionary for the neighbours
        (each key represents a node, and the corresponding value is a list of the node's neighbours)'''
    neighbours_dictionary = {}
    for node in G.nodes():
        neighbours = []
        for neighbor in G[node]:
            neighbours.append(neighbor)
        neighbours_dictionary[node] = neighbours
    return neighbours_dictionary

def get_adjacency(G):
    '''Converts the graph into a dictionary with each node as a key
       and its row in the adjacency matrix as its value'''
    nodes = list(G.nodes())
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_dictionary = {}
    for node in nodes:
        adjacency_dictionary[str(node)] = np.array(adjacency_matrix[nodes.index(node),:].todense()).flatten()
    return adjacency_dictionary
