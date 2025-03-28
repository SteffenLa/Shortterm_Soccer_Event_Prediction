"""
This file contains a Gaussian Naive Bayes classifier model that simply get the (possibly multivariate) time
series as input as suggested by the supervisors.
"""
from models.abstract import ClassificationModel
from sklearn.naive_bayes import GaussianNB


class NBClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(NBClassifier, self).__init__(model_config, random_state)
        self.model = GaussianNB(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.model.get_params()

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        # very naive
        return True

    @staticmethod
    def input_format():
        return "concat"
