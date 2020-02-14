import keras_metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from imblearn.keras import balanced_batch_generator
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

        print('SAMPLER', self.sampler_name)
        if self.sampler_name != '':
            # imbalance sampling
            training_generator, steps_per_epoch = \
                balanced_batch_generator(self.x_train, self.y_train, sampler=self.sampling.get_sampler(),
                                         batch_size=self.batch_size, random_state=self.random_seed)

            # training
            model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=self.epochs)

        else:
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)

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
            # imbalance sampling
            training_generator, steps_per_epoch = \
                balanced_batch_generator(self.x_train, self.y_train, sampler=self.sampling.get_sampler(),
                                         batch_size=self.batch_size, random_state=self.random_seed)

            # training
            model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=self.epochs)

        else:
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs)

        self.model = model
        print(model.summary())

    def test(self):
        # load model + test
        y_prob = self.model.predict(self.x_test)
        y_pred = y_prob.argmax(axis=-1)

        # evaluation metrics
        precision, recall, f1, accuracy = self.__evaluation(y_pred)

        return precision, recall, f1, accuracy
