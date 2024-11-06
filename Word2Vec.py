import numpy as np
import tensorflow as tf
import tqdm
import string
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import nltk
import os

logging.basicConfig(level=logging.DEBUG)


class TextRetriever(object):
    """ Utility class to read corpus """

    @staticmethod
    def standardize_text(input_text):
        return re.sub("[%s]" % re.escape(string.punctuation), "", input_text.lower())

    @staticmethod
    def tf_standardize_text(input_text):
        lowercase = tf.strings.lower(input_text)
        return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

    @staticmethod
    def read_file(path_to_file):
        vocab_size = 0
        with open(path_to_file, "r") as f:
            vocab_size = len(set(f.read().lower().split()))
        return path_to_file, vocab_size

    @staticmethod
    def write_corpus_file(corpus_name, dirname):
        corpus = getattr(nltk.corpus, corpus_name)
        files = corpus.fileids()
        filename = os.path.join(dirname, f"{corpus_name}.txt")
        with open(filename, "w") as fp:
            for f in files:
                words = corpus.words(f)
                for i in range(0, len(words), 10):
                    fp.write(' '.join(words[i:min(i+10, len(words))]) + '\n')
        return filename


    @staticmethod
    def read_corpus(corpus_name, dirname, sequence_len, batch_size, vocab_size=None):
        file_name = TextRetriever.write_corpus_file(corpus_name, dirname)
        path_to_file, vsz = TextRetriever.read_file(path_to_file=file_name)
        if vocab_size is None:
            vocab_size = vsz
        text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
        vectorize_layer = tf.keras.layers.TextVectorization(standardize=TextRetriever.tf_standardize_text, max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_len)
        vectorize_layer.adapt(text_ds.batch(batch_size))
        # returns vocabulary sorted in descending order by frequency
        text_vector_ds = text_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
        sequences = list(text_vector_ds.as_numpy_iterator())
        inverse_vocab = vectorize_layer.get_vocabulary()
        TextRetriever.inspect_dataset(sequences, inverse_vocab, 10)
        return sequences, inverse_vocab, vocab_size

    @staticmethod
    def inspect_dataset(sequences, inverse_vocab, num_to_inspect):
        logging.info(len(sequences))
        end = min(num_to_inspect, len(sequences))
        for seq in sequences[:end]:
            logging.info(f"{seq} => {[inverse_vocab[i] for i in seq]}")


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
            plt.savefig(os.path.join(dirname, f"ReducedWts_Word2Vec.jpeg"), dpi=500)
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


class Word2Vec(tf.keras.Model):
    """ Skipgram model """

    def __init__(self, embedding_dim, num_neg_samples, window_size, corpus_name, dirname,
                 batch_size=1024, seed=10, vocab_size=None, sequence_len=10,
                 buffer_size=10000):
        super(Word2Vec, self).__init__()
        self.dirname = dirname

        self.sequences, self.inverse_vocab, self.vocab_size = TextRetriever.read_corpus(corpus_name, dirname, sequence_len, batch_size, vocab_size)
        self.embedding_dim = embedding_dim
        self.num_neg_samples = num_neg_samples
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.word_to_index_dict = {v: i for i, v in enumerate(self.inverse_vocab)}
        self.target_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=1, name="target_emb")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=num_neg_samples + 1, name="context_softmax_emb")
        self.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)
        dotprod = tf.einsum("ik,ijk->ij", word_embed, context_embed)
        return dotprod

    def generate_training_data(self):
        """
        Generates skip-gram pairs with negative sampling for a list of sequences
        (int-encoded sentences) based on window size, number of negative samples
        and vocabulary size.
        """
        targets, contexts, labels = [], [], []
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.vocab_size)
        for sequence in tqdm.tqdm(self.sequences):
            positive_skipgrams, _ = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=self.vocab_size, sampling_table=sampling_table, window_size=self.window_size, negative_samples=0)
            for target_word, context_word in positive_skipgrams:
                context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
                neg_samples, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=self.num_neg_samples, unique=True, range_max=self.vocab_size, seed=self.seed, name="neg_sampling")
                context = tf.concat([tf.squeeze(context_class, 1), neg_samples], 0)
                label = tf.constant([1] + [0] * self.num_neg_samples, dtype="int64")
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return np.array(targets), np.array(contexts), np.array(labels)

    def fit(self, epochs=20):
        targets, contexts, labels = self.generate_training_data()
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
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


if __name__ == "__main__":
    embedding_dim = 128
    num_neg_samples = 5
    window_size = 3
    corpus_name = "reuters"
    vocab_size = 1024
    sequence_len = 10
    dirname = r"C:\prog\cygwin\home\samit_000\latex\book_stats\code\data"
    word2vec = Word2Vec(embedding_dim, num_neg_samples, window_size, corpus_name, dirname,
                        vocab_size=vocab_size,
                        sequence_len=sequence_len)
    word2vec.fit()
    weights_file = os.path.join(dirname, "weights.csv")
    word2vec.write_weights(weights_file)

    weights, words = word2vec.get_weights(top_n=10)
    Plotter.plot_weights(weights, size=10, dirname=dirname)
    logging.info(",".join(words))
    logging.info(weights)

    Word2Vec.get_similar_words(weights_file, topN=5)