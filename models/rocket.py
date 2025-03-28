"""
Rocket Time Series Classifier from sktime library
-> kernel-based
"""

from sktime.classification.kernel_based import RocketClassifier as RClassifier
from models.abstract import ClassificationModel


class RocketClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(RocketClassifier, self).__init__(model_config, random_state)
        self.model = RClassifier(**model_config)

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
