import csv
import os
import sys
from imbalance_anomaly.embedding.scikit_embedding import ScikitEmbedding
from imbalance_anomaly.methods.scikit_methods import ScikitModel


class Experiment(object):
    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method
        self.sampler = ['random-over-sampler', 'adasyn', 'smote', 'svm-smote',
                        'random-under-sampler', 'tomek-links', 'near-miss', 'instance-hardness']

    def __get_scikit_embedding(self):
        scikit_embedding = ScikitEmbedding(self.dataset)
        train_embedding, train_label, test_embedding, test_label = scikit_embedding.get_scikit_embedding()

        return train_embedding, train_label, test_embedding, test_label

    def __get_evaluation_file(self):
        current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', self.dataset))
        evaluation_file = os.path.join(current_path, self.method + '.evaluation.csv')

        return evaluation_file

    def run_scikit_methods(self):
        train_embedding, train_label, test_embedding, test_label = self.__get_scikit_embedding()

        evaluation_file = self.__get_evaluation_file()
        f = open(evaluation_file, 'wt')
        writer = csv.writer(f)
        for sampler in self.sampler:
            method = ScikitModel(train_embedding, train_label, test_embedding, test_label, self.method, sampler)
            precision, recall, f1, accuracy = method.run()
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
        experiment.run_scikit_methods()
