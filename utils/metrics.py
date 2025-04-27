import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, precision_score, recall_score, \
    balanced_accuracy_score, matthews_corrcoef


def apply_metrics(y_true, y_out, mode="classification"):
    if mode == "classification":
        return {
            "precision": precision_score(y_true, y_out),
            "recall": recall_score(y_true, y_out),
            "f1": f1_score(y_true, y_out),
            "accuracy": accuracy_score(y_true, y_out),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_out),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_out)
        }
    elif mode == "regression":
        return {
            "mse": mean_squared_error(y_true, y_out)
        }
    else:
        raise ValueError(f"Metric mode {mode} is not supported.")


def find_best_threshold(y_true, y_out, num_steps, metric="balanced_accuracy"):
    metric_dict = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    metric = metric_dict.get(metric)
    best_th = 0.0
    best_score = 0.0
    for i in np.linspace(0, 1, num_steps):
        if metric(y_true, y_out > i) > best_score:
            best_th = i
            best_score = metric(y_true, y_out > i)
    return best_th, best_score
