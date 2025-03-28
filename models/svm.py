"""
This file contains a SVM classifier and regression model that simply get the (possibly multivariate) time
series as input as suggested by the supervisors.
"""
from models.abstract import ClassificationModel, Model
from sklearn.svm import SVC, SVR


class SVMClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(SVMClassifier, self).__init__(model_config, random_state)
        self.model = SVC(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        num_limit = 50000
        if len(x) > num_limit:
            x = x[:num_limit]
            y = y[:num_limit]
            print(f"Cut number of training samples to {num_limit}")
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


class SVMRegressor(Model):
    def __init__(self, model_config, random_state):
        super(SVMRegressor, self).__init__(model_config, random_state)
        self.model = SVR(**model_config)

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
