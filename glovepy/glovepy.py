import re, gzip, time
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle
import threading
import numpy as np
import logging
import scipy
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from multiprocessing import Queue, Lock, cpu_count
from collections import OrderedDict
from .glove_inner import train_glove


try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle


class Glove(object):

    def __init__(self, cooccurence, alpha=0.75, x_max=100.0, d=50, seed=1234):
        """
        Glove model for obtaining dense embeddings from a
        co-occurence (sparse) matrix.
        """
        self.alpha           = alpha
        self.x_max           = x_max
        self.d               = d
        self.cooccurence     = cooccurence
        self.seed            = seed
        np.random.seed(seed)
        self.W               = np.random.uniform(-0.5/d, 0.5/d, (cooccurence.shape[0], d)).astype(np.float64)
        self.W_              = np.random.uniform(-0.5/d, 0.5/d, (cooccurence.shape[0], d)).astype(np.float64)
        self.b               = np.random.uniform(-0.5/d, 0.5/d, (cooccurence.shape[0], 1)).astype(np.float64)
        self.b_              = np.random.uniform(-0.5/d, 0.5/d, (cooccurence.shape[0], 1)).astype(np.float64)
        self.gradsqW         = np.ones_like(self.W, dtype=np.float64)
        self.gradsqW_        = np.ones_like(self.W_, dtype=np.float64)
        self.gradsqb         = np.ones_like(self.b, dtype=np.float64)
        self.gradsqb_        = np.ones_like(self.b_, dtype=np.float64)
        self.word2id         = None
        self.id2word         = None
        self.word_vectors    = None


    def train(self, step_size=0.05, workers=1, batch_size=100):
        """
        One epoch run to fit the word embedding matrix.
        """
        logger = logging.getLogger(__name__)

        jobs = Queue(maxsize=2 * workers)
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)
        total_error = [0.0]
        total_done  = [0]

        total_els = len(self.cooccurence.row)

        # Worker function:
        def worker_train():
            error = np.zeros(1, dtype = np.float64)
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                train_glove(self, job, step_size, error)
                with lock:
                    total_error[0] += error[0]
                    total_done[0] += len(job[0])
                    if total_done[0] % 10000 == 0:
                        logger.debug("Completed %.3f%%\r" % (100.0 * total_done[0] / total_els))
                error[0] = 0.0

        # Create workers
        workers_threads = [threading.Thread(target=worker_train) for _ in range(workers)]
        for thread in workers_threads:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # Batch co-occurence pieces
        batch_length = 0
        batch_row = []
        batch_col = []
        batch_data = []
        num_examples = 0

        for i in range(total_els):
            batch_row.append(self.cooccurence.row[i])
            batch_col.append(self.cooccurence.col[i])
            batch_data.append(self.cooccurence.data[i])
            batch_length += 1
            if batch_length >= batch_size:
                jobs.put(
                    (
                        np.array(batch_row, dtype=np.int32),
                        np.array(batch_col, dtype=np.int32),
                        np.array(batch_data, dtype=np.float64)
                    )
                )
                num_examples += len(batch_row)
                batch_row = []
                batch_col = []
                batch_data = []
                batch_length = 0

        # The very last batch
        if len(batch_row) > 0:
            jobs.put(
                (
                    np.array(batch_row, dtype=np.int32),
                    np.array(batch_col, dtype=np.int32),
                    np.array(batch_data, dtype=np.float64)
                )
            )
            num_examples += len(batch_row)
            batch_row = []
            batch_col = []
            batch_data = []
            batch_length = 0

        for _ in range(workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers_threads:
            thread.join()

        logger.info("Total Error = %.4f" % total_error[0])
        return total_error[0] / num_examples


    def add_dict(self, dictionary):
        """
        Add a word-to-id dictionary to the model and get its inverse dictionary for similarity query
        """
        logger = logging.getLogger(__name__)

        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')
        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller or equal to the number of word vectors')
        
        self.word2id = dictionary
        logger.debug("Added word-to-id dictionary")

        if hasattr(self.word2id, 'iteritems'):
            # Python 2 compat
            items_iterator = dictionary.iteritems()
        else:
            items_iterator = dictionary.items()
        self.id2word = {v: k for k, v in items_iterator}
        logger.debug("Added id-to-word dictionary")


    def fit(self, epochs=1, no_threads=1, step_size=0.05, batch_size=100):
        """
        Fit a Glove model with multiple epochs
        """
        logger = logging.getLogger(__name__)

        if no_threads > cpu_count():
            logger.warning("Threads number is more than CPU cores: %d, limited to the number of cores" % (cpu_count()))
            use_cores = cpu_count()
        else:
            use_cores = no_threads

        if self.cooccurence is None:
            raise Exception('Model must be provided a cooccurence matrix')
        
        errors = []

        for epoch in range(epochs):
            err = self.train(step_size=step_size, workers=use_cores, batch_size=batch_size)
            errors.append(err)
            logger.info("Epoch %d, average error %.4f" % (epoch, err))
        
        self.word_vectors = self.W
        return errors


    def get_norm(self, word_vec, ord=2):
        """
        Get the norm of the word vectors
        """
        logger = logging.getLogger(__name__)
        if ord is not None:
            logger.info("Precomputing L%d-norms of word weight vectors" % ord)
        else:
            logger.info("Default precomputing L2-norms of word weight vectors")

        if ord is not None and ord == 2:
            ord = None # The is the default setting in numpy to indicate L2 norm
        vec = word_vec.T / np.linalg.norm(word_vec, ord, axis=1)
        return vec.T

    
    def _similarity_query(self, word_vec, topn):
        """
        Function used to calculate the words distances and get the most similar words
        """
        dst = (np.dot(self.W, word_vec)
               / np.linalg.norm(self.W, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)
        return [(self.id2word[x], dst[x]) for x in word_ids[:(topn+1)]]


    def similar_words(self, word, topn=5):
        """
        Run a similarity query, retriving topn most similar words
        """
        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')
        if self.word2id is None:
            raise Exception('No word dictionary supplied')
        
        try:
            word_idx = self.word2id[word]
        except KeyError:
            raise Exception('Word not in dictionary')
        return self._similarity_query(self.word_vectors[word_idx], topn)[1:]


    def save(self, filename):
        """
        Serialize model to filename
        """
        with open(filename, 'wb') as savefile:
            pickle.dump(self.__dict__, 
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)
    

    @classmethod
    def load(cls, filename):
        """
        Deserialize model from filename
        """
        instance = Glove(cooccurence=np.array([[]]))

        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)
        return instance
        

    def unitvec(self, vec):
        """
        Scale a vector to unit length by applying L2-Norm.
        if the input is a zero vector, it will return back unchanged.
        """
        if scipy.sparse.issparse(vec):
            vec = vec.tocsr()
            veclen = np.sqrt(np.sum(vec.data ** 2))
            if veclen > 0.0:
                return vec / veclen
            else:
                return vec
        
        if isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=float)
            veclen = np.linalg.norm(vec)
            if veclen > 0.0:
                return vec / veclen
            else:
                return vec
        
        try:
            first = next(iter(vec)) # Is there at least one element?
        except:
            return vec
        raise ValueError("unknown input type")


    def most_similar(self, positive=[], negative=[], topn=5, word_norm=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.
        `model.word_vectors` must be a matrix of word embeddings (need to be L2-normalized),
        and its format must be either 2d numpy (dense) or scipy.sparse.csr.
        """
        logger = logging.getLogger(__name__)
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if isinstance(positive, str) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, (str, np.ndarray)) else word for word in positive]
        negative = [
            (word, -1.0) if isinstance(word, (str, np.ndarray)) else word for word in negative]
        
        # Use L2-norms of the word weight vectors if provided
        if word_norm is not None:
            logger.info("Provided normalized word embedding matrix")
            word_vec = word_norm
        else:
            logger.info("Using default word embedding matrix")
            word_vec = self.word_vectors

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            elif word in self.word2id:
                word_index = self.word2id[word]
                mean.append(weight * word_vec[word_index])
                all_words.add(word_index)
            else:
                logger.warning("word '%s' not in vocabulary" % word)
        
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        if scipy.sparse.issparse(word_vec):
            mean = scipy.sparse.vstack(mean)
        else:
            mean = np.array(mean)
        mean = self.unitvec(mean.mean(axis=0)).astype(word_vec.dtype)
        dists = word_vec.dot(mean.T).flatten()
        if not topn:
            return dists

        # Add more candidates in case the input words were in the results
        best = np.argsort(dists)[::-1][:topn + len(all_words)]

        # ignore (don't return) words from the input
        result = [(self.id2word[sim], float(dists[sim])) for sim in best if sim not in all_words][:topn]
        return OrderedDict(result)


    def word_vec(self, word, word_vec):
        """
        Get the word embedding vector by giving a word.
        """
        if word in self.word2id:
            return word_vec[self.word2id[word]]
        else:
            raise KeyError("word '%s' not in vocabulary" % word)


    def most_similar_cosmul(self, positive=[], negative=[], topn=5, word_norm=None):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg. Positive words still contribute positive,
        negative words contribute negatively towards the similarity, but with less
        susceptibility to one large distance dominating the calculation.

        `model.word_vectors` must be a matrix of word embeddings (need to be L2-normalized),
        or a normalized matrix word_norm needs to be provided.

        Original Paper:
        Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.
        """
        logger = logging.getLogger(__name__)
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if isinstance(positive, str) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
        
        all_words = set([self.word2id[word] for word in positive + negative
            if not isinstance(word, np.ndarray) and word in self.word2id])

        # Use L2-norms of the word weight vectors if provided
        if word_norm is not None:
            logger.info("Provided normalized word embedding matrix")
            word_vec = word_norm
        else:
            logger.info("Using default normalized word embedding matrix")
            word_vec = self.word_vectors
        
        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            self.word_vec(word, word_vec) if isinstance(word, str) else word for word in positive]
        negative = [
            self.word_vec(word, word_vec) if isinstance(word, str) else word for word in negative]
        
        if not positive:
            raise ValueError("cannot compute similarity with no input")
        
        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + np.dot(word_vec, term)) / 2) for term in positive]
        logger.debug("Positive distance: %s" % pos_dists)
        neg_dists = [((1 + np.dot(word_vec, term)) / 2) for term in negative]
        logger.debug("Negative distance: %s" % neg_dists)
        dists = np.prod(pos_dists, axis=0) / (np.prod(neg_dists, axis=0) + 0.000001)
        if not topn:
            return dists

        # Add more candidates in case the input words were in the results
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.id2word[sim], float(dists[sim])) for sim in best if sim not in all_words][:topn]
        return OrderedDict(result)


    def doesnt_match(self, words, word_norm=None):
        """
        Find out the word that does NOT similar to all other words in the given list.
        """
        logger = logging.getLogger(__name__)
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')
        
        good_words = [word for word in words if word in self.word2id]
        if len(good_words) != len(words):
            ignored_words = set(words) - set(good_words)
            logger.warning("Vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not good_words:
            raise ValueError("cannot select a word from an empty list")
        
        # Use L2-norms of the word weight vectors if provided
        if word_norm is not None:
            word_vec = word_norm
        else:
            word_vec = self.word_vectors
        
        vectors = np.vstack(self.word_vec(word, word_vec) for word in good_words).astype(word_vec.dtype)
        mean = self.unitvec(vectors.mean(axis=0)).astype(word_vec.dtype)
        dists = np.dot(vectors, mean)
        return sorted(zip(dists, good_words))[0][1]
        


