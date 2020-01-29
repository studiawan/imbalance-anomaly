from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss, InstanceHardnessThreshold
from imbalance_anomaly.imbalance.sampling import Sampling


class ScikitModel(object):
    def __init__(self, train_data, train_label, test_data, test_label, sampler=None):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.sampler = sampler
        self.random_seed = 101
        self.njobs = -1

    def __get_sampler(self):
        sampler = None
        if self.sampler == 'random-over-sampler':
            sampler = RandomOverSampler(random_state=self.random_seed)

        elif self.sampler == 'adasyn':
            sampler = ADASYN(random_state=self.random_seed)

        elif self.sampler == 'smote':
            sampler = SMOTE(random_state=self.random_seed)

        elif self.sampler == 'svm-smote':
            sampler = SVMSMOTE(random_state=self.random_seed)

        elif self.sampler == 'random-under-sampler':
            sampler = RandomUnderSampler(random_state=self.random_seed)

        elif self.sampler == 'tomek-links':
            sampler = TomekLinks()

        elif self.sampler == 'near-miss':
            sampler = NearMiss()

        elif self.sampler == 'instance-hardness':
            sampler = InstanceHardnessThreshold(random_state=self.random_seed, n_jobs=self.njobs)

        return sampler

    def __get_classifier(self, method):
        classifier = None
        if method == 'logistic-regression':
            classifier = LogisticRegression(random_state=self.random_seed, n_jobs=-self.njobs)

        elif method == 'svm':
            classifier = LinearSVC(random_state=self.random_seed)

        elif method == 'decision-tree':
            classifier = DecisionTreeClassifier(random_state=self.random_seed)

        elif method == 'passive-aggressive':
            classifier = PassiveAggressiveClassifier(random_state=self.random_seed, n_jobs=self.njobs)

        elif method == 'naive-bayes':
            classifier = GaussianNB()

        return classifier

    def __evaluation(self, predicted_label):
        precision, recall, f1, _ = precision_recall_fscore_support(self.test_label, predicted_label, average='macro')
        accuracy = accuracy_score(self.test_label, predicted_label)

        return precision, recall, f1, accuracy

    def run(self, method):
        # get classifier and sampler
        classifier = self.__get_classifier(method)
        sampler = self.__get_sampler()

        if self.sampler is not None:
            # sample the data
            train_resample, train_label_resample = sampler.fit_resample(self.train_data, self.train_label)

            # run classification
            classifier.fit(train_resample, train_label_resample)

        else:
            classifier.fit(self.train_data, self.train_label)

        # predict
        predicted_label = classifier.predict(self.test_data)

        # evaluation
        precision, recall, f1, accuracy = self.__evaluation(predicted_label)

        return precision, recall, f1, accuracy
