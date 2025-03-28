from abc import ABC, abstractmethod
from joblib import dump, load


class Model(ABC):

    @abstractmethod
    def __init__(self, model_config, random_state):
        self.model_config = model_config
        self.random_state = random_state

    @abstractmethod
    def train(self, x, y, x_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def save(self, path):
        dump(self, path)
        print("Model saved at path:", path)

    @staticmethod
    def load_from_path(path):
        return load(path)

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def is_parallelizable(self):
        pass

    @abstractmethod
    def is_multivariate(self):
        pass

    @abstractmethod
    def prediction_type(self):
        """
        :return: Either 'classification' or 'regression'
        """
        pass

    @staticmethod
    @abstractmethod
    def input_format():
        """
        :return: "time", "time_series" or "concat"
        """
        pass


class ClassificationModel(Model, ABC):

    @abstractmethod
    def predict_proba(self, x):
        pass

    def prediction_type(self):
        return "classification"
