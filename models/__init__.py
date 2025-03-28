#from .canonicalforest import CanonicalForestClassifier
#from .conv import ConvClassifier
#from .fullyconnected import NNClassifier
from .kneighbors import KNNRegressor, KNNClassifier, KNNTimeSeriesClassifier
from .linear import LogisticRegressionClassifier, LinearRegressor
#from .lstm import LSTMClassifier
#from .muse import MUSEClassifier
from .naive_bayes import NBClassifier
from .random import RandomGuesser
from .randomforest import RandomForestClassifier, RandomForestRegressor
#from .rocket import RocketClassifier
from .svm import SVMClassifier, SVMRegressor

models = {
    ##  "ConvClassifier": ConvClassifier, ## NN long, but not better
    ##  "LSTMClassifier": LSTMClassifier, ## NN long, but not better
    "KNNRegressor": KNNRegressor,  ##regression basierend auf KNN, not useful, as long as classification
    "RandomForestRegressor": RandomForestRegressor,
    "SVMRegressor": SVMRegressor,
    "LinearRegressor": LinearRegressor,
    ##  "MUSEClassifier": MUSEClassifier,
    ##"CanonicalForestClassifier": CanonicalForestClassifier,
    ##   "RocketClassifier": RocketClassifier,    ##timeseries relevant, but useless
    ##  "KNNTimeSeriesClassifier": KNNTimeSeriesClassifier,
    "KNNClassifier": KNNClassifier,
    #"NNClassifier": NNClassifier,   ## NN , only one to use, to check NN, but others are same problems
    "LogisticRegressionClassifier": LogisticRegressionClassifier,
    "NBClassifier": NBClassifier,   ##naive bayes
    "RandomForestClassifier": RandomForestClassifier,
    "SVMClassifier": SVMClassifier,
    "RandomGuesser": RandomGuesser ##random to check accuracy
}
