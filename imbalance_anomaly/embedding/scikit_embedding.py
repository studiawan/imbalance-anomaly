import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class ScikitEmbedding(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.glove_file = 'glove.6B.50d.txt'
        self.glove_dictionary = {}
        self.__read_embedding()
        self.train_label = {}
        self.test_label = {}

        self.MAX_PAD = 8
        self.GLOVE_DIM = 50
        self.MAX_NUM_WORDS = 400000
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
        current_path = \
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset, 'train-test'))
        train_file = os.path.join(current_path, self.dataset + '.train.pickle')
        test_file = os.path.join(current_path, self.dataset + '.test.pickle')

        with open(train_file, 'rb') as f:
            train_dict = pickle.load(f)

        with open(test_file, 'rb') as f:
            test_dict = pickle.load(f)

        return train_dict, test_dict

    def __pad_sequences(self):
        # read data
        train_dict, test_dict = self.__read_data()

        # train data
        train_list = []
        for line_id, properties in train_dict.items():
            self.train_label[line_id] = properties['label']
            train_list.append(properties['message'])

        # pad train data
        train_pad = pad_sequences(train_list, dtype=object, maxlen=self.MAX_PAD, value='_pad_',
                                  padding='post', truncating='post')

        # test data
        test_list = []
        for line_id, properties in test_dict.items():
            self.test_label[line_id] = properties['label']
            test_list.append(properties['message'])

        # pad test data
        test_pad = pad_sequences(test_list, dtype=object, maxlen=self.MAX_PAD, value='_pad_',
                                 padding='post', truncating='post')

        return train_pad, test_pad

    def get_scikit_embedding(self):
        train_pad, test_pad = self.__pad_sequences()
        train_embedding = []
        test_embedding = []

        # glove embedding for train data
        for message in train_pad:
            message_vectors = []
            for word in message:
                vectors = self.glove_dictionary.get(word, self.padding_vectors)
                message_vectors.extend(vectors)

            train_embedding.append(message_vectors)

        # glove embedding for test data
        for message in test_pad:
            message_vectors = []
            for word in message:
                vectors = self.glove_dictionary.get(word, self.padding_vectors)
                message_vectors.extend(vectors)

            test_embedding.append(message_vectors)

        return train_embedding, self.train_label, test_embedding, self.test_label
