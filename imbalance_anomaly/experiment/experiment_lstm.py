import os
import sys
import csv
from imbalance_anomaly.embedding.keras_embedding import KerasEmbedding
from imbalance_anomaly.methods.lstm import LSTMModel


class Experiment(object):
    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method
        self.sampler = ['random-over-sampler', 'adasyn', 'smote', 'svm-smote',
                        'random-under-sampler', 'tomek-links', 'near-miss', 'instance-hardness']

    def __get_evaluation_file(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        evaluation_file = os.path.join(current_path, self.method + '.evaluation.csv')

        return evaluation_file

    def __get_embedding(self):
        keras_embedding = KerasEmbedding(self.dataset)
        x_train, y_train, x_test, y_test, word_index, embedding_matrix = keras_embedding.get_data_and_embedding()

        return x_train, y_train, x_test, y_test, word_index, embedding_matrix

    def run_lstm(self):
        x_train, y_train, x_test, y_test, word_index, embedding_matrix = self.__get_embedding()

        evaluation_file = self.__get_evaluation_file()
        f = open(evaluation_file, 'wt')
        writer = csv.writer(f)
        for sampler in self.sampler:
            lstm_model = LSTMModel(x_train, y_train, x_test, y_test, word_index, embedding_matrix, sampler)
            lstm_model.train()
            precision, recall, f1, accuracy = lstm_model.test()
            writer.writerow([self.dataset, self.method, sampler, precision, recall, f1, accuracy])

        f.close()


if __name__ == '__main__':
    dataset_list = ['dfrws-2009', 'hofstede', 'secrepo']
    method_list = ['logistic-regression', 'svm', 'decision-tree', 'passive-aggressive', 'naive-bayes']

    if len(sys.argv) < 3:
        print('Please input dataset and method name.')
        print('python experiment.py dataset_name method_name')
        print('Supported datasets:', dataset_list)
        print('Supported methods :', method_list)
        sys.exit(1)

    else:
        dataset_name = sys.argv[1]
        method_name = sys.argv[2]
        experiment = Experiment(dataset_name, method_name)
        experiment.run_lstm()
