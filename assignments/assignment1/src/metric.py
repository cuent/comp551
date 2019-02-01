import numpy as np


def mse(y, y_pred):
    return np.square(y - y_pred).mean()
