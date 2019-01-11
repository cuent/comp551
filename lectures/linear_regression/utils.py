import numpy as np
from sys import platform as sys_pf

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def generate_data(f, n, sd=7):
    x = np.arange(n)
    y_ = f(x) + np.random.normal(0, sd, n)
    return x, y_


def plot_data(x, y, y_pred):
    f, ax = plt.subplots()
    ax.scatter(x, y, s=1, c='b', label="Real intances")
    ax.scatter(x, y_pred, s=1, c='r', label="Predicted instaces")
    return f, ax
