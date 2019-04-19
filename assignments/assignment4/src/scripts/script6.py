from src.data import MNIST
from src.models.model6 import VAE
from src.experiments.experiment6 import execute
import src.utils as utils

import logging
from pathlib import Path
import os
import torch.optim as optim
from torchvision.utils import save_image
import torch


def main(n, h, lr, noise_factor, noise_type, epochs, bs, exp_dir):
    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    # Logging
    log_dir = Path(exp_dir / 'logs/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(message)s',
                        handlers=[
                            logging.FileHandler(log_dir / 'experiment6'),
                            logging.StreamHandler()
                        ])

    # Configurations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    logging.info('n={},h={},lr={},noise_factor={},noise_type={},epochs={},batch_size={},dest={},device={}'
                 .format(n, h, lr, noise_factor, noise_type, epochs, bs, exp_dir, device))
    torch.manual_seed(590238490)
    train_loader = MNIST.train(bs)
    test_loader = MNIST.test(bs)
    model = VAE(h, n, noise_factor, noise_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ---------------------------------------------------------------
    # Execute experiment
    # ---------------------------------------------------------------
    execute(model, optimizer, epochs, train_loader, test_loader, device, n, exp_dir)


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------
    ns = [3, 5, 10, 20, 200]
    lrs = [1e-3] * 5
    epochs = 100
    bs = 100
    h = 500

    s_p = 'salt_and_pepper'
    noise_sp = .05

    # lr = 1e-3
    # noise_gauss = .25
    # gauss = 'gauss'

    # ---------------------------------------------------------------
    # Execute experiments
    # ---------------------------------------------------------------
    exp_dir = Path('../../data/results/experiment6/')
    exp_dir.mkdir(exist_ok=True)
    for n, lr in zip(ns, lrs):
        main(n=n, h=h, lr=lr, noise_factor=noise_sp, noise_type=s_p, epochs=epochs, bs=bs, exp_dir=exp_dir)
        utils.save_fig(exp_dir / "data_n_{}.pk".format(n), 'MNIST', n)
