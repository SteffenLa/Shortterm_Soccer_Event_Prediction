"""
KNeighborsTimeSeriesClassifier from sktime library
-> distance-based
"""

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from models.abstract import ClassificationModel, Model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNNTimeSeriesClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(KNNTimeSeriesClassifier, self).__init__(model_config, random_state)
        self.model = KNeighborsTimeSeriesClassifier(**model_config)

    def train(self, x, y, x_val=None, y_val=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.model.get_params()

    def is_parallelizable(self):
        return False

    def is_multivariate(self):
        return True

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    @staticmethod
    def input_format():
        return "concat"


class KNNClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(KNNClassifier, self).__init__(model_config, random_state)
        self.model = KNeighborsClassifier(**model_config)

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


class KNNRegressor(Model):
    def __init__(self, model_config, random_state):
        super(KNNRegressor, self).__init__(model_config, random_state)
        self.model = KNeighborsRegressor(**model_config)

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
