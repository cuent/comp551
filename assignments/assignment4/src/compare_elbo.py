from sys import platform as sys_pf
import os
import pickle
import numpy as np
from pathlib import Path

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_training(exp_folder, name, i, n):
    cmap = plt.get_cmap('jet_r')
    x1, y1, x2, y2 = pickle.load(open(exp_folder / "data_n_{}.pk".format(n), "rb"))
    color = cmap(float(i) / 5)

    # Training
    plt.plot(x1, -y1, label='train {}'.format(name), c=color)

    # Testing
    # plt.plot(x2, -y2, label='test {}'.format(name).format(n), linestyle='--', c=color)


def main(n, base_dir, exps, names):
    fig = plt.figure()

    for i, info in enumerate(zip(exps, names)):
        plot_training(base_dir / info[0], info[1], i, n)

    plt.xlabel('Training samples')
    plt.ylabel('$\mathcal{L}$')
    plt.legend(loc='lower right')
    plt.title("Log-likelihood ($N_z$={})".format(n))

    elbo_dir = Path(base_dir / 'elbo')
    elbo_dir.mkdir(exist_ok=True)
    plt.savefig(elbo_dir / 'elbo_n_{}.png'.format(n))


def main_plot_all(ns, base_dir, exps, names):
    fig = plt.figure(figsize=(15, 6))

    for j, n in enumerate(ns):
        plt.subplot(1, 5, j + 1)
        for i, info in enumerate(zip(exps, names)):
            plot_training(base_dir / info[0], info[1], i, n)

        if j == 2:
            plt.xlabel('Training samples')
        if j == 0:
            plt.ylabel('$\mathcal{L}$')
        if j == 4:
            plt.legend(loc='lower right')
        plt.title("Log-likelihood ($N_z$={})".format(n))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,4))
    # fig.legend(lines, labels, loc=(0.5, 0), ncol=5)
    elbo_dir = Path(base_dir / 'elbo')
    elbo_dir.mkdir(exist_ok=True)
    plt.savefig(elbo_dir / 'elbo_all.png')


if __name__ == '__main__':
    ns = [3, 5, 10, 20, 200]
    base_dir = Path('../data/results/')
    exps = [
        'experiment1_choose_lr',
        'experiment1_relu',
        'experiment3',
        # 'experiment4',
        # 'experiment6'
    ]
    names = [
        'VAE',
        'VAE-tuned',
        'VAE CNN',
        # 'IWAE',
        # 'DVAE'
    ]
    # for n in ns:
    #     main(n, base_dir, exps, names)

    main_plot_all(ns, base_dir, exps, names)
