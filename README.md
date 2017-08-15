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

### Build on Windows

If you are installing this package on Windows (tested on Windows 10), you will need to install Visual C++ 2015 Build Tools. If you have already installed Microsoft Visual Studio, you cannot install the Visual C++ Build Tools. It will ask you to uninstall your existing VS when you tried to install the Visual C++ build tools using the standalone installer. However, since you already have the VS, you can go to Control Panel—Programs and Features and right click the VS item and Change-Modify, then check the option of those components that relates to the Visual C++ Build Tools, like Visual C++, Windows SDD, then install them. Please also make sure the Build tools path has been added to the system path.

Once you have the Visual C++ Build Tools, you can build the package as follows:

```python
python setup.py build_ext --inplace --compiler=msvc
python setup.py install
```

### Linux

Install from the source code:

```python
python setup.py install
```
