import numpy as np
import os
import gensim
from tqdm import tqdm
import random

from utils import get_graph, get_neighbours

def get_random_walks_for_one_node(node,neighbours,walks_per_node,steps):
    '''Performs several random walks from a given node'''
    walks = []
    for i in range(0,walks_per_node):
        current_node = node
        walk = [current_node]
        for step in range(1,steps):
            current_node = np.random.choice(neighbours[current_node], 1)[0]
            walk.append(current_node)
        walks.append(walk)
    return walks

def get_random_walks(neighbours, walks_per_node, steps):
    ''' This function returns a list of random walks '''
    walks = []
    for node in tqdm(neighbours):
        walks += get_random_walks_for_one_node(node,neighbours,walks_per_node,steps)
    random.shuffle(walks)
    return walks

def get_embeddings(input_file, output_folder, directed=False, walks_per_node=10, steps=80, 
                   size=300, window=10, workers=1, verbose=True):
    """
    Performs uniform random walks on given graph and generates its embeddings.

    :param input_file: Path to a file containing an edge list of a graph (str). 
    :param output_folder: Directory where the embeddings will be stored (str).
    :param directed: True if the graph is directed (bool).
    :param walks_per_node: How many random walks will be performed from each node (int).
    :param steps: How many node traversals will be performed for each random walk (int).
    :param size: Base dimensionality of the embedding vector. Should be divisable by 6 (int).
    :param window: The window parameter for the word2vec model (i.e. maximum distance in a random walk where one node can be considered the another node's context) (int).
    :param workers: Number of threads to use when training the word2vec model (int).
    :param verbose: Whether to print progress messages to stdout (bool).
    """
    
    if verbose:
        print("Getting the graph")
    graph = get_graph(input_file, directed)
    
    if verbose:
        print("Getting the neighbours' dictionary")
    neighbours = get_neighbours(graph)
    
    if verbose:
        print("Getting the random walks")
    random_walks = get_random_walks(neighbours, walks_per_node, steps)
    
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


    
