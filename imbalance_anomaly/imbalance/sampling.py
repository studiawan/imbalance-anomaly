from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss, InstanceHardnessThreshold


class Sampling(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.random_seed = 101
        self.njobs = -1

    def get_sampler(self):
        sampler = None
        if self.sampler == 'random-over-sampler':
            sampler = RandomOverSampler(random_state=self.random_seed)

        elif self.sampler == 'adasyn':
            sampler = ADASYN(random_state=self.random_seed, n_jobs=self.njobs)

        elif self.sampler == 'smote':
            sampler = SMOTE(random_state=self.random_seed, n_jobs=self.njobs)

        elif self.sampler == 'svm-smote':
            sampler = SVMSMOTE(random_state=self.random_seed, n_jobs=self.njobs)

        elif self.sampler == 'random-under-sampler':
            sampler = RandomUnderSampler(random_state=self.random_seed)

        elif self.sampler == 'tomek-links':
            sampler = TomekLinks(n_jobs=self.njobs)

        elif self.sampler == 'near-miss':
            sampler = NearMiss(n_jobs=self.njobs)

        elif self.sampler == 'instance-hardness':
            sampler = InstanceHardnessThreshold(random_state=self.random_seed, n_jobs=self.njobs)

        return sampler
