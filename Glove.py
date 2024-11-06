import numpy as np
import tensorflow as tf
import tqdm
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from src.Word2Vec import TextRetriever
import os

logging.basicConfig(level=logging.DEBUG)


class WeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, xmax=100.0, power=0.75):
        super().__init__()
        self.xmax = xmax
        self.power = power

    def call(self, y_true, y_pred):
        xij = tf.exp(y_true) - 1
        sample_weight = tf.where(xij < self.xmax, tf.pow(xij / self.xmax, self.power), 1)
        val = tf.math.square(y_true - y_pred)
        return tf.math.reduce_mean(sample_weight * val)


class Glove(tf.keras.Model):
    """ Word co-occurrence model """

    def __init__(self, embedding_dim, window_size, corpus_name, dirname, xmax=100, power=0.75,
                 skip_words=None, batch_size=1024, seed=10, vocab_size=None, sequence_len=10,
                 buffer_size=10000):
        super(Glove, self).__init__()
        self.dirname = dirname

        self.sequences, self.inverse_vocab, self.vocab_size = TextRetriever.read_corpus(corpus_name, dirname, sequence_len, batch_size, vocab_size)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.skip_words = {}
        if skip_words:
            self.skip_words = set(skip_words)
        self.co_occurrence_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int32)
        self.word_to_index_dict = {v: i for i, v in enumerate(self.inverse_vocab)}
        self.target_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, name="target_emb")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, name="context_emb")
        self.bias_a = tf.keras.layers.Embedding(vocab_size, 1, input_length=1, name="bias1")
        self.bias_b = tf.keras.layers.Embedding(vocab_size, 1, input_length=1, name="bias2")
        self.compile(optimizer="adam", loss=WeightedMeanSquaredError(xmax=xmax, power=power), metrics=["accuracy"])

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        if len(context.shape) == 2:
            context = tf.squeeze(context, axis=1)
        word_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)
        bias_a = tf.squeeze(self.bias_a(target), axis=1)
        bias_b = tf.squeeze(self.bias_b(context), axis=1)
        dotprod = tf.einsum("ij,ij->i", word_embed, context_embed)
        return dotprod + bias_a + bias_b

    def generate_training_data(self):
        """
        Generates skip-gram pairs for a list of sequences
        (int-encoded sentences) based on window size, number of negative samples
        and vocabulary size.
        """
        targets, contexts = [], []
        self.co_occurrence_matrix[:, :] = 0
        for sequence in tqdm.tqdm(self.sequences):
            positive_skipgrams, _ = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=self.vocab_size, window_size=self.window_size)
            for target_word, context_word in positive_skipgrams:
                if (self.inverse_vocab[target_word] in self.skip_words) or (
                        self.inverse_vocab[context_word] in self.skip_words):
                    continue
                self.co_occurrence_matrix[target_word, context_word] += 1
                self.co_occurrence_matrix[context_word, target_word] += 1
                targets.append(target_word)
                contexts.append(context_word)

        return np.array(targets), np.array(contexts)

    def fit(self, epochs=20):
        targets, contexts = self.generate_training_data()
        output = np.log(self.co_occurrence_matrix + 1)
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), output[targets, contexts]))
        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        super().fit(dataset, epochs=epochs)

    def write_weights(self, file_name):
        weights = self.target_embedding.get_weights()[0]
        with open(file_name, "w") as fp:
            for index, word in enumerate(self.inverse_vocab):
                if index == 0:
                    continue  # skip 0, it's padding.
                vec = weights[index]
                fp.write(word + "," + ",".join([str(x) for x in vec]) + "\n")

    def get_weights(self, top_n=None, word_list=None):
        if top_n:
            word_list = self.inverse_vocab[1:top_n + 1]
        weights = self.target_embedding.get_weights()[0]
        indices = np.array([self.word_to_index_dict.get(w, 0) for w in word_list])
        return weights[indices, :], word_list

    def cosine_similarity(self, top_n=None, word_list=None):
        if top_n:
            word_list = self.inverse_vocab[1:top_n + 1]
        nwords = len(word_list)
        weights = self.target_embedding.get_weights()[0]
        indices = np.array([self.word_to_index_dict.get(w, 0) for w in word_list])
        wts = weights[indices, :]
        lengths = np.sum(np.multiply(wts, wts), axis=1)
        cosine = np.zeros((nwords, nwords), dtype=np.float64)
        for i in range(nwords):
            cosine[i, i] = 1.0
            for j in range(i):
                cosine[i, j] = np.dot(wts[i, :], wts[j, :]) / np.sqrt(lengths[i] * lengths[j])
                cosine[j, i] = cosine[i, j]
        return cosine, word_list

    def length_similarity(self, top_n=None, word_list=None):
        if top_n:
            word_list = self.inverse_vocab[1:top_n + 1]
        nwords = len(word_list)
        weights = self.target_embedding.get_weights()[0]
        indices = np.array([self.word_to_index_dict.get(w, 0) for w in word_list])
        wts = weights[indices, :]

        lengths = np.zeros((nwords, nwords), dtype=np.float64)
        for i in range(nwords):
            for j in range(i):
                dist = np.subtract(wts[i, :], wts[j, :])
                lengths[i, j] = np.sqrt(np.dot(dist, dist))
                lengths[j, i] = lengths[i, j]
        return lengths, word_list

    @staticmethod
    def get_similar_words(weights_file, topN=6):
        np.random.seed(64)
        df = pd.read_csv(weights_file, header=None)
        words = np.random.choice(df.shape[0], 10, replace=False)
        similarWords = [[]]
        for i in range(topN):
            similarWords.append([])

        for iword in words:
            word = df.loc[iword, 0]
            vec = df.loc[iword, 1:].values
            l1 = np.dot(vec, vec)
            cosineArr = []
            for j in range(df.shape[0]):
                if j == iword:
                    continue
                word2 = df.loc[j, 0]
                vec2 = df.loc[j, 1:].values
                l2 = np.dot(vec2, vec2)
                cosineSim = np.dot(vec, vec2) / np.sqrt(l1 * l2)
                cosineArr.append((cosineSim, word2))
            cosineArr.sort(key=lambda x: x[0], reverse=True)
            similarWords[0].append(word)
            for i in range(topN):
                similarWords[i+1].append(cosineArr[i][1])

        columns = ["word"] + ["SimWord%d" % (i+1) for i in range(topN)]
        data = {c:arr for c,arr in zip(columns, similarWords)}
        df2 = pd.DataFrame(data=data)
        logging.info(df2.to_latex(index=False))


class Plotter(object):
    @staticmethod
    def plot_weights(weights, size, labels=None, dirname=None):
        if size < weights.shape[1]:
            weights = Plotter.reduce_to_k_dim(weights, size)
        if labels is None:
            labels = ["%d" % (i + 1) for i in range(size)]

        data = pd.DataFrame(weights, columns=labels)
        pd.plotting.scatter_matrix(data, alpha=0.2, diagonal='hist', figsize=(10, 10))
        if dirname:
            plt.savefig(os.path.join(dirname, f"ReducedWts_Glove.jpeg"), dpi=500)
        plt.show()

    @staticmethod
    def reduce_to_k_dim(M, k=2, n_iter=10):
        """ Reduce a matrix M (n, m) to a matrix of dimensionality (n, k) using the
            following SVD function from Scikit-Learn:
                - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

            Params:
                M (n,m): co-occurence matrix of word counts
                k (int): embedding size of each word after dimension reduction
            Return:
                M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                        In terms of the SVD from math class, this actually returns U * S
        """
        svd = TruncatedSVD(n_components=k, n_iter=n_iter)
        return svd.fit_transform(M)


if __name__ == "__main__":
    embedding_dim = 128
    window_size = 3
    corpus_name = "reuters"
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    vocab_size = 1024
    sequence_len = 10
    glove = Glove(embedding_dim, window_size, corpus_name, dirname, vocab_size=vocab_size, sequence_len=sequence_len)
    glove.fit(epochs=2)
    weights_file = os.path.join(dirname, "weights_glove.csv")
    glove.write_weights(weights_file)

    weights, words = glove.get_weights(top_n=10)
    Plotter.plot_weights(weights, size=10, dirname=dirname)
    logging.info(",".join(words))
    logging.info(weights)

    Glove.get_similar_words(weights_file, topN=5)