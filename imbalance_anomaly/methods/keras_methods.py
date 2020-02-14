import keras_metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical
from imbalance_anomaly.imbalance.sampling import Sampling


class KerasModel(object):
    def __init__(self, x_train, y_train, x_test, y_test, word_index, embedding_matrix, sampler_name=''):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.sampler_name = sampler_name
        self.model = None
        self.dropout = 0.4
        self.units = 256
        self.activation = 'tanh'
        self.batch_size = 128
        self.epochs = 20
        self.random_seed = 101
        self.MAX_PAD = 8
        self.GLOVE_DIM = 50
        self.sampling = Sampling(sampler_name)
        self.sampler = self.sampling.get_sampler()
        self.filters = 64
        self.kernel_size = 5

    def __evaluation(self, predicted_label):
        true_label = self.y_test.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(true_label, predicted_label, average='macro')
        accuracy = accuracy_score(true_label, predicted_label)

        return precision*100, recall*100, f1*100, accuracy*100

    def train_lstm(self):
        # build model and compile
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    self.GLOVE_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_PAD,
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        model.add(SpatialDropout1D(self.dropout))
        model.add(LSTM(self.units, dropout=self.dropout, recurrent_dropout=self.dropout, activation=self.activation))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(), 'acc'])

        if self.sampler_name != '':
            # sample the data
            train_resample, train_label_resample = self.sampler.fit_resample(self.x_train, self.y_train)
            train_resample = np.asarray(train_resample)
            train_label_resample = to_categorical(train_label_resample)

            # training
            model.fit(train_resample, train_label_resample, batch_size=self.batch_size, epochs=self.epochs)

        else:
            x_train = np.asarray(self.x_train)
            y_train = to_categorical(self.y_train)
            model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

        self.model = model
        print(model.summary())

    def train_cnn(self):
        # build model and compile
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    self.GLOVE_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_PAD,
                                    trainable=False)
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(self.dropout))
        model.add(Conv1D(self.filters, self.kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        # model.add(Dropout(self.dropout))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        if self.sampler_name != '':
            # sample the data
            train_resample, train_label_resample = self.sampler.fit_resample(self.x_train, self.y_train)
            train_resample = np.asarray(train_resample)
            train_label_resample = to_categorical(train_label_resample)

            # training
            model.fit(train_resample, train_label_resample, batch_size=self.batch_size, epochs=self.epochs)

        else:
            x_train = np.asarray(self.x_train)
            y_train = to_categorical(self.y_train)
            model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

        self.model = model
        print(model.summary())

    def test(self):
        # load model + test
        y_prob = self.model.predict(self.x_test)
        y_pred = y_prob.argmax(axis=-1)

        # evaluation metrics
        precision, recall, f1, accuracy = self.__evaluation(y_pred)

        return precision, recall, f1, accuracy
