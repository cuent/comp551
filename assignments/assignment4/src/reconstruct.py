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
from src.models.model7 import VAE as CVAE
from src.models.model8 import VAE as BetaVAE
from src.utils import idx2onehot


def get_ordered_digits():
    data = MNIST.test(64, dir='../data')

    digits_tensors = []
    digits_labels = []
    found = []
    for x, labels in data:
        for i, digit in enumerate(labels):
            if digit.item() not in found:
                found.append(digit.item())
                digits_tensors.append(x[i])
                digits_labels.append(digit)
            elif len(found) == 10:
                break
        if len(found) == 10:
            break

    torch_digits_tensors = torch.stack(digits_tensors)
    torch_digits_labels = torch.stack(digits_labels)
    id_sort = torch.argsort(torch.Tensor(digits_labels))

    # select digits in ascending order
    torch_digits_labels = torch_digits_labels[id_sort]
    torch_digits_tensors = torch_digits_tensors[id_sort]

    # Testing
    # for i, data in enumerate(torch_digits_tensors):
    #     img = data.numpy().transpose(1, 2, 0)
    #     lbl = torch_digits_labels[i]
    #     print(lbl)
    #     cv2.imshow('', img)
    #     cv2.waitKey()
    return torch_digits_tensors, torch_digits_labels


def exp1(base_dir, device, x, n):
    model = BVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_choose_lr/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    y, _, _ = model(x)
    return y


def exp1_1(base_dir, device, x, n):
    model = BVAE_RELU(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_relu/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    y, _, _ = model(x)
    return y


def exp3(base_dir, device, x, n):
    model = CNNVAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment3/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    y, _, _ = model(x)
    return y


def exp4(base_dir, device, x, n):
    model = IWAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment4/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    _, _, _, p = model(x)
    return p


def exp6(base_dir, device, x, n):
    noise_sp = .05
    s_p = 'salt_and_pepper'
    model = DVAE(500, n, noise_sp, s_p).to(device).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment6/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    x = x.to(device)
    y, _, _ = model(x)
    return y


def exp7(base_dir, device, x, c, n):
    model = CVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment7/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    c = idx2onehot(c, n=10)
    x, c = x.to(device), c.to(device)
    x = x.to(device)
    y, _, _ = model(x, c)

    return y


def exp8(base_dir, device, x, n):
    model = BetaVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment8/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    y, _, _ = model(x)

    return y


def reconstruct(data, labels, device, base_dir, latent):
    # ---------------------------------------------------------------
    # Get true data and reconstruction
    # ---------------------------------------------------------------
    n = data.shape[0]
    data = data.to(device)

    comparison = torch.cat([data,
                            exp1(base_dir, device, data, latent).view(n, 1, 28, 28),
                            exp1_1(base_dir, device, data, latent).view(n, 1, 28, 28),
                            exp3(base_dir, device, data, latent).view(n, 1, 28, 28),
                            exp4(base_dir, device, data, latent).view(n, 1, 28, 28),
                            exp6(base_dir, device, data, latent).view(n, 1, 28, 28),
                            exp7(base_dir, device, data, labels, latent).view(n, 1, 28, 28),
                            exp8(base_dir, device, data, latent).view(n, 1, 28, 28),
                            ])
    save_image(comparison.cpu(), base_dir / 'reconstruct/reconstruct_n_{}.png'.format(latent), nrow=n, padding=1,
               pad_value=1)


def main(n, dir='../data/results'):
    # ---------------------------------------------------------------
    # Setup Configurations
    # ---------------------------------------------------------------
    base_dir = Path(dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X, y = get_ordered_digits()

    # ---------------------------------------------------------------
    # Execute and save image
    # ---------------------------------------------------------------
    reconstruct(X, y, device, base_dir, n)


if __name__ == '__main__':
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        main(n)
