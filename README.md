# GlovePy

A Python implementation with Cython of the [GloVe](http://www-nlp.stanford.edu/projects/glove/) algorithm with multi-threaded training.
this Python package mainly contains two Python classes. The first Python class (Corpus) builds the co-occurrence matrix given a collection of
documents; while the second Python class (Glove) will generate vector representations for words.

GloVe is an unsupervised learning algorithm for generating vector representations for words developed by Stanford NLP lab.
The paper describing the model is [here](http://nlp.stanford.edu/projects/glove/glove.pdf). In contrast with
[Word2Vec](https://code.google.com/p/word2vec/) (there is a great Python implementation in [gensim](http://radimrehurek.com/gensim/models/word2vec.html))
which is often referred as prediction method, GloVe is called counting method which the embedding is produced by factorizing the logarithm of the
corpus word co-occurrence matrix.

The original implementation for this Machine Learning model can be [found here](http://nlp.stanford.edu/projects/glove/). This work is based on the work of [glove-python](https://github.com/maciejkula/glove-python) and [glove](https://github.com/JonathanRaiman/glove).

## Installation

Install from the source code: `python setup.py install`.

