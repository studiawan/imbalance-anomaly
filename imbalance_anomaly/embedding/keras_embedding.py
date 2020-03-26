import os
import numpy as np
import pickle
import random
from math import floor
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class KerasEmbedding(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.glove_file = 'glove.6B.50d.txt'
        self.glove_dictionary = {}
        self.RANDOM_SEED = 101
        self.TRAIN_SIZE = 0.7
        self.MAX_PAD = 8
        self.GLOVE_DIM = 50
        self.MAX_NUM_WORDS = 400000

    def __read_embedding(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'glove'))
        glove_path = os.path.join(current_path, self.glove_file)

        with open(glove_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], dtype='float32')
                self.glove_dictionary[word] = vectors

    def __read_data(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, 'log.all.pickle')

        with open(groundtruth_file, 'rb') as f:
            data = pickle.load(f)

        return data

    def __tokenize_pad(self, data_dict):
        # get data
        data_list = []
        data_label = []
        for line_id, properties in data_dict.items():
            data_label.append(properties['label'])
            data_list.append(' '.join(properties['message']))

        # tokenize
        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(data_list)
        sequences = tokenizer.texts_to_sequences(data_list)
        word_index = tokenizer.word_index

        # pad data
        data_pad = pad_sequences(sequences, maxlen=self.MAX_PAD, padding='post', truncating='post')

        return word_index, data_pad, data_label

    def __split(self, data_pad, labels):
        # check normal and anomaly
        normal_index = []
        anomaly_index = []
        for line_id, label in enumerate(labels):
            if label == 1:
                normal_index.append(line_id)
            elif label == 0:
                anomaly_index.append(line_id)

        # initialize train and test
        train_data = []
        train_label = []
        test_data = []
        test_label = []

        # random sequence for normal
        list_len = len(normal_index)
        random.Random(self.RANDOM_SEED).shuffle(normal_index)

        train_length = floor(self.TRAIN_SIZE * list_len)
        for index in normal_index[:train_length]:
            train_data.append(data_pad[index])
            train_label.append(labels[index])

        for index in normal_index[train_length:]:
            test_data.append(data_pad[index])
            test_label.append(labels[index])

        # random sequence for anomaly
        negative_len = len(anomaly_index)
        if negative_len > 0:
            random.Random(self.RANDOM_SEED).shuffle(anomaly_index)

            train_length = floor(self.TRAIN_SIZE * negative_len)
            for index in anomaly_index[:train_length]:
                train_data.append(data_pad[index])
                train_label.append(labels[index])

            for index in anomaly_index[train_length:]:
                test_data.append(data_pad[index])
                test_label.append(labels[index])

        return train_data, train_label, test_data, test_label

    def __get_embedding_matrix(self, word_index):
        # prepare embedding matrix
        num_words = min(self.MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.GLOVE_DIM))

        for word, i in word_index.items():
            if i >= self.MAX_NUM_WORDS:
                continue

            embedding_vector = self.glove_dictionary.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_data_and_embedding(self):
        self.__read_embedding()
        data = self.__read_data()

        # tokenize, pad, and split
        word_index, data_pad, labels = self.__tokenize_pad(data)
        x_train, y_train, x_test, y_test = self.__split(data_pad, labels)
        embedding_matrix = self.__get_embedding_matrix(word_index)

        return x_train, y_train, x_test, y_test, word_index, embedding_matrix
