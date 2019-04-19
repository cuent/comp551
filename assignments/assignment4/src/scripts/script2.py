from src.data import FreyFace
from src.models.model2 import VAE
from src.experiments.experiment2_fface import execute
import src.utils as utils

import logging
from pathlib import Path
import os
import torch.optim as optim
from torchvision.utils import save_image
import torch


def main(n, h, lr, epochs, bs, exp_dir):
    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    # Logging
    log_dir = Path(exp_dir / 'logs/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(message)s',
                        handlers=[
                            logging.FileHandler(log_dir / 'experiment2'),
                            logging.StreamHandler()
                        ])

    # Configurations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('n={},lr={},h={},epochs={},batch size={},dest={},device={}'
                 .format(n, lr, h, epochs, bs, exp_dir, device))
    torch.manual_seed(590238490)
    # device = 'cpu'
    train_loader = FreyFace.train(bs)
    test_loader = FreyFace.train(bs)
    model = VAE(h, n).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # ---------------------------------------------------------------
    # Execute experiment
    # ---------------------------------------------------------------
    execute(model, optimizer, epochs, train_loader, test_loader, device, n, exp_dir)


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------
    ns = [2, 5, 10, 20]  # latent space
    epochs = 3000
    bs = 100
    lr = 0.01
    h = 200

    # ---------------------------------------------------------------
    # Execute experiments
    # ---------------------------------------------------------------
    exp_dir = Path('../../data/results/experiment2/')
    exp_dir.mkdir(exist_ok=True, parents=True)
    for i, n in enumerate(ns):
        main(n=n, h=h, lr=lr, epochs=epochs, bs=bs, exp_dir=exp_dir)
        utils.save_fig(exp_dir / "data_n_{}.pk".format(n), 'MNIST', n)
