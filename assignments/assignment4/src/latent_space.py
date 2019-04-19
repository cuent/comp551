from src.data import MNIST
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
import pickle
from pathlib import Path
import logging
from src.models.model1 import VAE as BVAE
from src.models.model1_1 import VAE as BVAE_RELU
from src.models.model2 import VAE as GVAE
from src.models.model3 import VAE as CNNVAE
from src.models.model4 import VAE as IWAE
from src.models.model6 import VAE as DVAE
from mpl_toolkits.mplot3d import axes3d
from sys import platform as sys_pf

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def get_data():
    data = MNIST.test(4000, dir='../data')
    return data


def exp1(base_dir, device, x, n):
    x = x.view(-1, 28 * 28)
    model = BVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_choose_lr/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    z = model.sample(*model.encoder(x))
    return z


def exp1_1(base_dir, device, x, n):
    x = x.view(-1, 28 * 28)
    model = BVAE_RELU(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_relu/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    z = model.sample(*model.encoder(x))
    return z


def exp3(base_dir, device, x, n):
    model = CNNVAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment3/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    return model.sample(*model.encoder(x))


def exp4(base_dir, device, x, n):
    x = x.view(-1, 28 * 28)
    model = IWAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment4/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    _, _, _, h2, _, _ = model.encoder(x)
    return h2


def exp6(base_dir, device, x, n):
    x = x.view(-1, 28 * 28)
    noise_sp = .05
    s_p = 'salt_and_pepper'
    model = DVAE(500, n, noise_sp, s_p).to(device).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment6/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    return model.sample(*model.encoder(x))


def save_image(z, y, base_dir, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=y, cmap='brg')
    plt.title('Latent space of {} (n={})'.format(name, 3))

    latent_dir = Path(base_dir / 'latent')
    latent_dir.mkdir(exist_ok=True)
    plt.savefig(latent_dir / 'latent_{}_n_{}.png'.format(name, 3))


def latent_save(data, device, base_dir, latent):
    # ---------------------------------------------------------------
    # Get true data and reconstruction
    # ---------------------------------------------------------------
    X, y = iter(data).next()
    n = X.shape[0]
    X = X.to(device)
    y = y.numpy()

    # ---------------------------------------------------------------
    # VAE exp 1
    # ---------------------------------------------------------------
    z = exp1(base_dir, device, X, latent).detach().numpy()
    save_image(z, y, base_dir, "VAE")

    # ---------------------------------------------------------------
    # VAE tuned exp 1.1
    # ---------------------------------------------------------------
    z = exp1_1(base_dir, device, X, latent).detach().numpy()
    save_image(z, y, base_dir, "VAE-tuned")

    # ---------------------------------------------------------------
    # VAE CNN exp 3
    # ---------------------------------------------------------------
    z = exp3(base_dir, device, X, latent).detach().numpy()
    save_image(z, y, base_dir, "VAE-CNN")

    # ---------------------------------------------------------------
    # IWAE exp 4
    # ---------------------------------------------------------------
    z = exp4(base_dir, device, X, latent).detach().numpy()
    save_image(z, y, base_dir, "IWAE")

    # ---------------------------------------------------------------
    # DVAE exp 6
    # ---------------------------------------------------------------
    z = exp6(base_dir, device, X, latent).detach().numpy()
    save_image(z, y, base_dir, "DVAE")


def main(n, dir='../data/results'):
    # ---------------------------------------------------------------
    # Setup Configurations
    # ---------------------------------------------------------------
    base_dir = Path(dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = get_data()

    # ---------------------------------------------------------------
    # Execute and save image
    # ---------------------------------------------------------------
    latent_save(data, device, base_dir, n)


if __name__ == '__main__':
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        main(ns)
