import numpy as np
import os
import gensim
from tqdm import tqdm
import random

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from utils import get_graph, get_neighbours, get_adjacency

def get_similarities_common_neighbours(neighbours):
    """Returns the similarities for each node of every other node 
       given the common neighbors metric"""
    p_dictionary = {}
    for node_i in tqdm(neighbours):
        prob = []
        nod = []
        neighbours_i = set(neighbours[node_i])
        neighbours2 = [i for i in neighbours if i != node_i]
        for node_j in neighbours2:
            neighbours_j = set(neighbours[node_j])
            similarity = len(neighbours_i.intersection(neighbours_j))
            prob.append(similarity)
            nod.append(node_j)
        
        if sum(prob) != 0.0:
            prob = [x/float(sum(prob)) for x in prob]
        
        p_dictionary[node_i] = (nod,prob)

    return p_dictionary

def get_similarities_jaccard(neighbours):
    """Returns the similarities for each node of every other node 
       given the jaccard metric"""
    p_dictionary = {}
    for node_i in tqdm(neighbours):
        prob = []
        nod = []
        neighbours_i = set(neighbours[node_i])
        neighbours2 = [i for i in neighbours if i != node_i]
        for node_j in neighbours2:
            neighbours_j = set(neighbours[node_j])
            similarity = len(neighbours_i.intersection(neighbours_j))/float(len(neighbours_i.union(neighbours_j)))
            prob.append(similarity)
            nod.append(node_j)
        
        if sum(prob) != 0.0:
            prob = [x/float(sum(prob)) for x in prob]
        
        p_dictionary[node_i] = (nod,prob)

    return p_dictionary

def get_similarities_euclidean(neighbours, adjacency_dictionary):
    """Returns the similarities for each node of every other node 
       given the euclidean metric"""
    p_dictionary = {}
    for node_i in tqdm(neighbours):
        prob = []
        nod = []
        adj_i = adjacency_dictionary[node_i]
        neighbours2 = [i for i in neighbours if i != node_i]
        for node_j in neighbours2:
            adj_j = adjacency_dictionary[node_j]
            similarity = distance.euclidean(adj_i, adj_j)
            prob.append(similarity)
            nod.append(node_j)

        if sum(prob) != 0.0:
            prob_min, prob_max = min(prob), max(prob)
            if (prob_max-prob_min) != 0.0:
                prob = [(i-prob_min)/(prob_max-prob_min) for i in prob]
            
            prob = [(1-i) for i in prob]

            prob = [x/float(sum(prob)) for x in prob]
        
        p_dictionary[node_i] = (nod,prob)
    return p_dictionary

def get_similarities_cosine(neighbours, adjacency_dictionary):
    """Returns the similarities for each node of every other node 
       given the cosine similarity"""
    p_dictionary = {}
    for node_i in tqdm(neighbours):
        prob = []
        nod = []
        adj_i = adjacency_dictionary[node_i].reshape(1, -1)
        neighbours2 = [i for i in neighbours if i != node_i]
        for node_j in neighbours2:
            adj_j = adjacency_dictionary[node_j].reshape(1, -1)
            similarity = cosine_similarity(adj_i,adj_j)[0][0]
            prob.append(similarity)
            nod.append(node_j)

        if sum(prob) != 0.0:
            prob = [x/float(sum(prob)) for x in prob]
        
        p_dictionary[node_i] = (nod,prob)
    return p_dictionary

def get_similarities_pearson(neighbours, adjacency_dictionary):
    """Returns the similarities for each node of every other node 
       given the pearson metric"""
    p_dictionary = {}
    for node_i in tqdm(neighbours):
        prob = []
        nod = []
        adj_i = adjacency_dictionary[node_i]
        neighbours2 = [i for i in neighbours if i != node_i]
        for node_j in neighbours2:
            adj_j = adjacency_dictionary[node_j]
            similarity = abs(pearsonr(adj_i, adj_j)[0])
            prob.append(similarity)
            nod.append(node_j)
        
        if sum(prob) != 0.0:
            prob = [x/float(sum(prob)) for x in prob]
        
        p_dictionary[node_i] = (nod,prob)
    return p_dictionary

def get_random_walks_for_one_node(node, neighbours, similarities, walks_per_node, steps):
    '''Performs several random walks from a given node with the traversal probabilities 
       given by a dictionary'''
    walks = []
    for i in range(0,walks_per_node):
        current_node = node
        walk = [current_node]
        for step in range(1,steps):
            if sum(similarities[current_node][1]) == 1.0:
                current_node = np.random.choice(similarities[current_node][0], p=similarities[current_node][1])
            else:
                current_node = np.random.choice(neighbours[current_node])
            walk.append(current_node)
        walks.append(walk)
    return walks

def get_random_walks(neighbours, similarities, walks_per_node, steps):
    ''' This function returns a list of random walks '''
    walks = []
    for node in tqdm(neighbours):
        walks += get_random_walks_for_one_node(node, neighbours, similarities, walks_per_node, steps)
    random.shuffle(walks)
    return walks


def get_embeddings(input_file, output_folder, directed=False, walks_per_node=10, steps=80,
                   size=300, window=10, workers=1, metric='jaccard', verbose=True):  
    """
    Performs non-uniform random walks (on neighboring nodes) on given graph and generates its embeddings.

    :param input_file: Path to a file containing an edge list of a graph (str). 
    :param output_folder: Directory where the embeddings will be stored (str).
    :param directed: True if the graph is directed (bool).
    :param walks_per_node: How many random walks will be performed from each node (int).
    :param steps: How many node traversals will be performed for each random walk (int).
    :param size: Dimensionality of the embedding vector. Should be divisable by 6  (int).
    :param window: The window parameter for the word2vec model (i.e. maximum distance in a random walk where one node can be considered the another node's context) (int).
    :param workers: Number of threads to use when training the word2vec model (int).
    :param metric: The metric which will be used to generate similarities (str).
    :param verbose: Whether to print progress messages to stdout (bool).
    """

    if verbose:
        print("Getting the graph")
    graph = get_graph(input_file, directed)
        
    if verbose:
        print("Getting the neighbours' dictionary")
    neighbours = get_neighbours(graph)
    
    if verbose:
        print("Getting the similarities")

    if metric == "common_neighbours":
        similarities = get_similarities_common_neighbours(neighbours)
    elif metric == 'jaccard':
        similarities = get_similarities_jaccard(neighbours)
    elif metric == 'euclidean':
        adjacency_dictionary = get_adjacency(graph)
        similarities = get_similarities_euclidean(neighbours, adjacency_dictionary)
    elif metric == 'cosine':
        adjacency_dictionary = get_adjacency(graph)
        similarities = get_similarities_cosine(neighbours, adjacency_dictionary)
    elif metric == 'pearson':
        adjacency_dictionary = get_adjacency(graph)
        similarities = get_similarities_pearson(neighbours, adjacency_dictionary)
    else:
        raise ValueError("Invalid value for parameter 'metric'.\n" + \
                         "Should be one of: 'common_neighbours', 'jaccard', 'euclidean', 'cosine', 'pearson'")
    if verbose:
        print("Getting the random walks")
    random_walks = get_random_walks(neighbours, similarities, walks_per_node, steps)

    if verbose:
        print("Getting the embeddings")
        print(size)
    model = gensim.models.Word2Vec(random_walks, min_count=0, size=size, window=window, iter=1, sg=1, workers=workers)
    model.wv.save_word2vec_format(os.path.join(output_folder, 'embeddings_' + str(size) + '.csv'))
    
    if verbose:
        print(int(size/2))
    model = gensim.models.Word2Vec(random_walks, min_count=0, size=int(size/2), window=window, iter=1, sg=1, workers=workers)
    model.wv.save_word2vec_format(os.path.join(output_folder, 'embeddings_' + str(int(size/2)) + '.csv'))
    
    if verbose:
        print(int(size/3))
    model = gensim.models.Word2Vec(random_walks, min_count=0, size=int(size/3), window=window, iter=1, sg=1, workers=workers)
    model.wv.save_word2vec_format(os.path.join(output_folder, 'embeddings_' + str(int(size/3)) + '.csv'))

