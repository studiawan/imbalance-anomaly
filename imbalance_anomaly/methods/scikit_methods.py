from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from imbalance_anomaly.imbalance.sampling import Sampling


class ScikitModel(object):
    def __init__(self, train_data, train_label, test_data, test_label, method, sampler_name=''):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.method = method
        self.sampler_name = sampler_name
        self.random_seed = 101
        self.njobs = -1
        self.sampling = Sampling(sampler_name)
        self.sampler = self.sampling.get_sampler()

    def __get_classifier(self):
        classifier = None
        if self.method == 'logistic-regression':
            classifier = LogisticRegression(random_state=self.random_seed, n_jobs=-self.njobs)

        elif self.method == 'svm':
            classifier = LinearSVC(random_state=self.random_seed)

        elif self.method == 'decision-tree':
            classifier = DecisionTreeClassifier(random_state=self.random_seed)

        elif self.method == 'passive-aggressive':
            classifier = PassiveAggressiveClassifier(random_state=self.random_seed, n_jobs=self.njobs)

        elif self.method == 'naive-bayes':
            classifier = GaussianNB()

        return classifier

    def __evaluation(self, predicted_label):
        precision, recall, f1, _ = precision_recall_fscore_support(self.test_label, predicted_label, average='macro')
        accuracy = accuracy_score(self.test_label, predicted_label)

        return precision*100, recall*100, f1*100, accuracy*100

    def run(self):
        # get classifier and sampler
        classifier = self.__get_classifier()

        if self.sampler_name != '':
            # sample the data
            train_resample, train_label_resample = self.sampler.fit_resample(self.train_data, self.train_label)

            # run classification
            classifier.fit(train_resample, train_label_resample)

        else:
            classifier.fit(self.train_data, self.train_label)

        # predict
        predicted_label = classifier.predict(self.test_data)

        # evaluation
        precision, recall, f1, accuracy = self.__evaluation(predicted_label)

        return precision, recall, f1, accuracy
