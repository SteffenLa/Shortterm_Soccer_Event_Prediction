"""
Canonical Interval Forest Classifier from sktime library
-> interval-based
"""

from sktime.classification.interval_based import CanonicalIntervalForest
from models.abstract import ClassificationModel


class CanonicalForestClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(CanonicalForestClassifier, self).__init__(model_config, random_state)
        self.model = CanonicalIntervalForest(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.model.get_params()

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        return True

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    @staticmethod
    def input_format():
        return "time_series"
