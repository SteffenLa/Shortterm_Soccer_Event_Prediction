"""
This file contains a random forest classifier and regression model that simply get the (possibly multivariate) time series as input.
"""
from models.abstract import ClassificationModel, Model
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR


class RandomForestClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(RandomForestClassifier, self).__init__(model_config, random_state)
        self.model = RFC(**model_config)

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


class RandomForestRegressor(Model):
    def __init__(self, model_config, random_state):
        super(RandomForestRegressor, self).__init__(model_config, random_state)
        self.model = RFR(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.model.get_params()

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        # very naive
        return True

    def prediction_type(self):
        return "regression"

    @staticmethod
    def input_format():
        return "concat"
