import os
import pickle
import numpy as np
import random
from math import floor
from keras.preprocessing.sequence import pad_sequences


class ScikitEmbedding(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.glove_file = 'glove.6B.50d.txt'
        self.glove_dictionary = {}
        self.__read_embedding()

        self.MAX_PAD = 8
        self.GLOVE_DIM = 50
        self.MAX_NUM_WORDS = 400000
        self.RANDOM_SEED = 101
        self.TRAIN_SIZE = 0.7
        self.padding_vectors = [0.] * self.GLOVE_DIM

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
        groundtruth_file = os.path.join(current_path, 'auth.all.pickle')

        with open(groundtruth_file, 'rb') as f:
            data = pickle.load(f)

        return data

    def __pad_sequences(self, data):
        # data
        data_list = []
        labels = []
        for line_id, properties in data.items():
            labels[line_id] = properties['label']
            data_list.append(properties['message'])

        # pad train data
        data_pad = pad_sequences(data_list, dtype=object, maxlen=self.MAX_PAD, value='_pad_',
                                 padding='post', truncating='post')

        return data_pad, labels

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

    def get_scikit_embedding(self):
        # get data, padding, and split
        data = self.__read_data()
        data_pad, labels = self.__pad_sequences(data)
        train_data, train_label, test_data, test_label = self.__split(data_pad, labels)

        # initialize embedding
        train_embedding = []
        test_embedding = []

        # glove embedding for train data
        for message in train_data:
            message_vectors = []
            for word in message:
                vectors = self.glove_dictionary.get(word, self.padding_vectors)
                message_vectors.extend(vectors)

            train_embedding.append(message_vectors)

        # glove embedding for test data
        for message in test_data:
            message_vectors = []
            for word in message:
                vectors = self.glove_dictionary.get(word, self.padding_vectors)
                message_vectors.extend(vectors)

            test_embedding.append(message_vectors)

        return train_embedding, train_label, test_embedding, test_label
