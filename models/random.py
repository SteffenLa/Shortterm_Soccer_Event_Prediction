"""
'Random' guessing baseline model.
Uses a binomial distribution with p = ratio of training samples with goals in training set.
Always predicts this ratio as probability when asked for predict_proba(x).
"""

from models.abstract import ClassificationModel
import numpy as np


class RandomGuesser(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(RandomGuesser, self).__init__(model_config, random_state)
        self.p = None

    def train(self, x, y, x_val=None, y_val=None):
        self.p = y.sum() / len(y)

    def predict(self, x):
        if self.p is not None:
            return np.random.binomial(1, self.p, len(x)).astype(bool)
        else:
            raise ValueError("train() needs to be called before predict()")

    def predict_proba(self, x):
        if self.p is not None:
            return self.p
        else:
            raise ValueError("train() needs to be called before predict()")

    def get_parameters(self):
        return {"p": self.p}

    def find_best_model(self, search_config):
        self.train(search_config.get("x"), search_config.get("y"), )
        print(f"Best model: p = {self.p}")

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        return True

    @staticmethod
    def input_format():
        return "concat"
