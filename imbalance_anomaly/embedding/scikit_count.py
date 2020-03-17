import os
import pickle
import random
from math import floor
from sklearn.feature_extraction.text import CountVectorizer


class ScikitCount(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.RANDOM_SEED = 101
        self.TRAIN_SIZE = 0.7

    def __read_data(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        groundtruth_file = os.path.join(current_path, 'log.all.pickle')

        with open(groundtruth_file, 'rb') as f:
            data = pickle.load(f)

        # data
        data_list = []
        labels = []
        for line_id, properties in data.items():
            labels.append(properties['label'])
            data_list.append(' '.join(properties['message']))

        return data_list, labels

    @staticmethod
    def __get_count_vectorizer(data):
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(data)

        return vectors.toarray()

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
        data, labels = self.__read_data()
        vectors = self.__get_count_vectorizer(data)
        train_data, train_label, test_data, test_label = self.__split(vectors, labels)

        return train_data, train_label, test_data, test_label
