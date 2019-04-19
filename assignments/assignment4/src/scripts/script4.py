from src.data import MNIST
from src.models.model4 import VAE
from src.experiments.experiment4 import execute
import src.utils as utils

import logging
from pathlib import Path
import os
import torch.optim as optim
from torchvision.utils import save_image
import torch


def main(n, k, lr, epochs, bs, exp_dir):
    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    # Logging
    log_dir = Path(exp_dir / 'logs/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(message)s',
                        handlers=[
                            logging.FileHandler(log_dir / 'experiment4'),
                            logging.StreamHandler()
                        ])

    # Configurations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    logging.info('n={},lr={},k={},epochs={},bs={},dest={},device={}'.format(n, lr, k, epochs, bs, exp_dir, device))
    torch.manual_seed(590238490)
    train_loader = MNIST.train(bs)
    test_loader = MNIST.test(bs)
    model = VAE(n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=10e-4)
    # ---------------------------------------------------------------
    # Execute experiment
    # ---------------------------------------------------------------
    execute(model, optimizer, epochs, train_loader, test_loader, device, n, k, exp_dir)


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------
    ns = [3, 5, 10, 20, 200]  # latent space
    lrs = [1e-3,1e-3,1e-3,1e-3,1e-3]
    epochs = 100
    bs = 200
    k = 50

    # ---------------------------------------------------------------
    # Execute experiments
    # ---------------------------------------------------------------
    exp_dir = Path('../../data/results/experiment4/')
    exp_dir.mkdir(exist_ok=True, parents=True)
    for n,lr in zip(ns, lrs):
        main(n=n, k=k, lr=lr, epochs=epochs, bs=bs, exp_dir=exp_dir)
        utils.save_fig(exp_dir / "data_n_{}.pk".format(n), 'MNIST', n)
