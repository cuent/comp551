from sys import platform as sys_pf
import os
import pickle
import numpy as np
import torch

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def save_fig(directory, name='MNIST', n=2):
    # cmap = plt.get_cmap('jet_r')
    # color = cmap(float(i) / len(ns))

    # # ---------------------------------------------------------------
    # Read data
    # ---------------------------------------------------------------
    x1, y1, x2, y2 = pickle.load(open(directory, "rb"))

    fig = plt.figure()

    # Training
    plt.subplot(2, 1, 1)
    plt.plot(x1, -y1, label='train $N_z=${}'.format(n), c='blue')
    plt.xlabel('training samples')
    plt.ylabel('$\mathcal{L}$')
    plt.legend(loc='lower right')
    plt.title(name)

    # Testing
    plt.subplot(2, 1, 2)
    plt.plot(x2, -y2, label='test $N_z=${}'.format(n), linestyle='--', c='orange')
    plt.xlabel('testing samples')
    plt.ylabel('$\mathcal{L}$')
    plt.legend(loc='lower right')
    plt.title(name)
    plt.savefig(directory.parent / 'bound_n_{}.png'.format(n))


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot
