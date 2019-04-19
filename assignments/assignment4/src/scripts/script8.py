from src.models.model8 import VAE
from src.experiments.experiment8 import MNIST_vae
from src.data import MNIST as mnist_

import torch.optim as optim
# from torchvision.utils import save_image
# import torch
# import pickle
import matplotlib.pyplot as plt
# import os
import numpy as np

from pathlib import Path
import logging

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s',
                    handlers=[
                        logging.FileHandler(log_dir / 'beta_vae'),
                        logging.StreamHandler()
                    ])

# Hyperparameters to change 
num_epochs = 100
batch_size = 100
learning_rate = 0.001
hidden_unit = 500


def main(n, h, epochs, bs, learning_rate):
    #    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = mnist_.train(bs)
    test_loader = mnist_.test(bs)
    model = VAE(h, n)
    #    model = model.to(device) # for Adagrad: https://github.com/pytorch/pytorch/issues/7321
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #    optimizer = optim.Adagrad(model.parameters(), lr = learning_rate)

    logging.info('n={},h={},lr={},epochs={},batch size={}'
                 .format(n, h, learning_rate, epochs, bs))

    exp = MNIST_vae(model, optimizer, learning_rate, bs, n, train_loader, test_loader)

    train_points, train_loss, test_points, test_loss, seq_train_loss, seq_test_loss = exp.execute(epochs)
    return train_points, train_loss, test_points, test_loss, seq_train_loss, seq_test_loss


if __name__ == '__main__':
    nz = [3, 5, 10, 20, 200]

    plt.figure(figsize=(10, 3))
    for i, n in enumerate(nz):
        print('_' * 20 + str(n) + '_' * 20)
        x1, y1, x2, y2, seq_train_loss, seq_test_loss = main(n, hidden_unit, num_epochs, batch_size, learning_rate)

        seq_train_loss = seq_train_loss.mean(axis=1)
        seq_test_loss = seq_test_loss.mean(axis=1)

        ax = plt.subplot(1, len(nz), i + 1)
        plt.plot(np.arange(0, len(seq_train_loss), 1), - seq_train_loss)
        plt.plot(np.arange(0, len(seq_test_loss), 1), - seq_test_loss)
        # plt.ylim(-140, -50)
        plt.title('Nz: {}'.format(n))
        plt.legend(['train', 'test'])
    plt.savefig('vae.png')
    plt.show()
