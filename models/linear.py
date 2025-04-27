from models.abstract import ClassificationModel, Model
from sklearn.linear_model import LogisticRegression, LinearRegression


class LogisticRegressionClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(LogisticRegressionClassifier, self).__init__(model_config, random_state)
        self.model = LogisticRegression(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.model.get_params()

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        # very naive
        return True

    @staticmethod
    def input_format():
        return "concat"


class LinearRegressor(Model):
    def __init__(self, model_config, random_state):
        super(LinearRegressor, self).__init__(model_config, random_state)
        self.model = LinearRegression(**model_config)

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
