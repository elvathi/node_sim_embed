import os
import argparse
import gensim
import numpy as np

import get_embeddings_uniform
import get_embeddings_similar_neighbours
import get_embeddings_similar_any

def parse_the_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--metric', type=str, default='jaccard')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_the_args()
    print('U')
    folder = os.path.join(args.output_folder, 'U')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(args.size) + '.csv')):
        get_embeddings_uniform.get_embeddings(args.input_file, folder, directed=args.directed, walks_per_node=args.walks_per_node, steps=args.steps, size=args.size, window=args.window, workers=args.workers)

    print('S_nbr')
    folder = os.path.join(args.output_folder, 'S_nbr')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(args.size) + '.csv')):
        get_embeddings_similar_neighbours.get_embeddings(args.input_file, folder, directed=args.directed, walks_per_node=args.walks_per_node, steps=args.steps, size=args.size, window=args.window, workers=args.workers, metric=args.metric)
        
    print('S_any')
    folder = os.path.join(args.output_folder, 'S_any')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(args.size) + '.csv')):
        get_embeddings_similar_any.get_embeddings(args.input_file, folder, directed=args.directed, walks_per_node=args.walks_per_node, steps=args.steps, size=args.size, window=args.window, workers=args.workers, metric=args.metric)
    
    embeddingsI = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'U', 'embeddings_' + str(int(args.size/2)) + '.csv'))
    embeddingsII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'S_nbr', 'embeddings_' + str(int(args.size/2)) + '.csv'))
    embeddingsIII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'S_any', 'embeddings_' + str(int(args.size/2)) + '.csv'))

    print('U+S_nbr')
    folder = os.path.join(args.output_folder, 'U+S_nbr')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv')):
        embeddings = gensim.models.keyedvectors.KeyedVectors(args.size)
        for node in embeddingsI.index2word:
            values = np.concatenate((embeddingsI[node],embeddingsII[node]))
            embeddings.add(node, values)
        embeddings.save_word2vec_format(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv'))

    print('U+S_any')
    folder = os.path.join(args.output_folder, 'U+S_any')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv')):
        embeddings = gensim.models.keyedvectors.KeyedVectors(args.size)
        for node in embeddingsI.index2word:
            values = np.concatenate((embeddingsI[node],embeddingsIII[node]))
            embeddings.add(node, values)
        embeddings.save_word2vec_format(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv'))

    print('S_nbr+S_any')
    folder = os.path.join(args.output_folder, 'S_nbr+S_any')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv')):
        embeddings = gensim.models.keyedvectors.KeyedVectors(args.size)
        for node in embeddingsI.index2word:
            values = np.concatenate((embeddingsII[node],embeddingsIII[node]))
            embeddings.add(node, values)
        embeddings.save_word2vec_format(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv'))
    
    del embeddingsI
    del embeddingsII
    del embeddingsIII
    
    print('U+S_nbr+S_any')
    folder = os.path.join(args.output_folder, 'U+S_nbr+S_any')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.isfile(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv')):
        embeddingsI = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'U', 'embeddings_' + str(int(args.size/3)) + '.csv'))
        embeddingsII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'S_nbr', 'embeddings_' + str(int(args.size/3)) + '.csv'))
        embeddingsIII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(args.output_folder, 'S_any', 'embeddings_' + str(int(args.size/3)) + '.csv'))
        embeddings = gensim.models.keyedvectors.KeyedVectors(args.size)
        for node in embeddingsI.index2word:
            values = np.concatenate((embeddingsI[node],embeddingsII[node],embeddingsIII[node]))
            embeddings.add(node, values)
        embeddings.save_word2vec_format(os.path.join(folder, 'embeddings_' + str(int(args.size)) + '.csv'))
        
