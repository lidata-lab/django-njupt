"""Metrics for regression tasks.

"""

import numpy as np

from mlv82.utils import to_numpy

__all__ = [
    "mean_squared_error",
    "root_mean_squared_error",
    "norm_root_mean_squared_error"
]

def mean_squared_error(true, pred):
    """Computes the mean squared error of prediction.

    Args:
        true: The true values of y.
        pred: The predicted values of y.

    Returns:
        The value of mse.
    """
    true_np = to_numpy(true)
    pred_np = to_numpy(pred)
    squared_error_np = (true_np - pred_np) ** 2
    return np.sum(squared_error_np) / true_np.size

def root_mean_squared_error(true, pred):
    """Computes the root mean squared error of prediction.

    Args:
        true:
        pred:

    Returns:
        The value of rmse.
    """
    mse = mean_squared_error(true, pred)
    return np.sqrt(mse)

def norm_root_mean_squared_error(true, pred, methods="std"):
    true_np = to_numpy(true)
    rmse = root_mean_squared_error(true, pred)
    if(methods == "std"):
        return rmse / np.std(true_np)
    elif(methods == "mean"):
        return rmse / np.mean(true_np)
    else:
        return rmse / (np.max(true_np) - np.min(true_np))

