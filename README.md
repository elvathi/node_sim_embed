### Repository of the paper:
## *Unsupervised Graph Embedding based on Node Similarity*
### by Eleni Vathi, Thanos Tagaris, Georgios Alexandridis and Andreas Stafylopatis


Abstract:

> In recent years graph embedding techniques have gained significant attention. Modern approaches benefit from the latest language models to encode a graph's vertices in a vector space, while retaining the information involving the structure of the graph.
The exploration of the graph is achieved through random walks, which are then treated as the equivalent of sentences. These approaches, however, do not take into account the similarities between the nodes when implementing these random walks.
In this work, an unsupervised approach for retrieving latent representations of vertices in a graph, is presented. Inspired by traditional community detection algorithms, which compute the similarity between each pair of vertices with respect to some property, the resulting representations not only encode the information about their adjacency in a vector space, but also take into account various similarities between the vertices.
The performance of the proposed methodology is evaluated on the task of multi-label classification, for several artificial and real-world networks. Results show that this methodology outperforms the state-of-the-art algorithms in networks with strong community structure.

### Requirements
All scripts were run with a python3 interpreter. The requirements are listed below:

* numpy (1.16.3)
* networkx (2.2)
* gensim (3.8.0)
* tqdm (4.32.2)


### Code Organization
The `source` directory provides an implementation of the methodology proposed in the paper. `main.py` is the main file to run the code and can be executed as follows:

`python ./source/main.py --input_file input_file --output_folder output_folder`

The rest of the scripts include:
* `get_embeddings_uniform.py`: An implementation of the "U" exploration procedure, as described in the paper.
* `get_embeddings_similar_neighbours.py`: An implementation of the "S<sub>nbr</sub>" exploration procedure, as described in the paper.
* `get_embeddings_similar_any.py`: An implementation of the "S<sub>any</sub>" exploration procedure, as described in the paper.
* `utils.py`: Contains various functions used by the previous scripts.

#### Input
The `input_file` is the path to an (unweighted) edgelist file.
Each line in the file represents an edge of the graph, and contains the IDs of the two nodes which form the edge:

`node_1_id node_2_id`

#### Output
The `output_folder` is the directory where the resulting files containing the embeddings will be saved  The `main.py` script creates six sub-directories in `output_folder` (named `U`, `S_nbr`, `S_any`, `U+S_nbr`, `U+S_any`, `S_nbr+S_any`, `U+S_nbr+S_any`), one for each of the six embedding strategies proposed in the paper.

Each sub-directory contains a file of the trained word vectors, in a format compatible with the original word2vec implementation, as they are stored by the [gensim](https://radimrehurek.com/gensim/index.html) framework (via self.wv.save_word2vec_format).

#### Optional arguments

Apart from the `input_file` and the `output_folder`, there are other optional arguments:
* `directed`: True if the graph is directed (bool).
* `walks_per_node`: How many random walks will be performed from each node (int).
* `steps`: How many node traversals will be performed for each random walk (int).
* `size`: Dimensionality of the embedding vector. Should be divisable by 6  (int).
* `window`: The window parameter for the word2vec model (i.e. maximum distance in a random walk where one node can be considered the context of another node) (int).
* `workers`: Number of threads to use when training the word2vec model (int).
* `metric`: The similarity metric used (str). Should be one of 'common_neighbours', 'jaccard', 'euclidean', 'cosine', 'pearson'.
* `verbose`: Whether to print progress messages to stdout (bool).

<!---
### Results

### Citing
-->
