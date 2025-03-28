"""
MUSE (MUltivariate Symbolic Extension) Classifier from sktime library
-> dictionary-based
"""

from sktime.classification.dictionary_based import MUSE
from models.abstract import ClassificationModel


class MUSEClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(MUSEClassifier, self).__init__(model_config, random_state)
        self.model = MUSE(**model_config)

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
