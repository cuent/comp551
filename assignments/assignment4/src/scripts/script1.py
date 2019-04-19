from src.data import MNIST
from src.models.model1 import VAE
from src.experiments.experiment1_mnist import execute
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
                            logging.FileHandler(log_dir / 'experiment1'),
                            logging.StreamHandler()
                        ])

    # Configurations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('n={},h={},lr={},epochs={},batch size={},dest={},device={}'
                 .format(n, h, lr, epochs, bs, exp_dir, device))
    torch.manual_seed(590238490)
    # device = 'cpu'
    train_loader = MNIST.train(bs)
    test_loader = MNIST.test(bs)
    model = VAE(h, n).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    # ---------------------------------------------------------------
    # Execute experiment
    # ---------------------------------------------------------------
    execute(model, optimizer, epochs, train_loader, test_loader, device, n, exp_dir)


if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------
    ns = [3, 5, 10, 20, 200]  # latent space
    lrs = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
    epochs = 100
    bs = 100
    h = 500

    # ---------------------------------------------------------------
    # Execute experiments
    # ---------------------------------------------------------------
    exp_dir = Path('../../data/results/experiment1_choose_lr/')
    exp_dir.mkdir(exist_ok=True)
    for lr, n in zip(lrs, ns):
        main(n=n, h=h, lr=lr, epochs=epochs, bs=bs, exp_dir=exp_dir)
        utils.save_fig(exp_dir / "data_n_{}.pk".format(n), 'MNIST', n)
