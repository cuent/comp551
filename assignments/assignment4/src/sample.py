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


def generate_sample(n, device):
    return torch.randn(10, n).to(device)


def exp1(base_dir, device, x, n):
    model = BVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_choose_lr/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    return model.decoder(x).cpu()


def exp1_1(base_dir, device, x, n):
    model = BVAE_RELU(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment1_relu/models/experiment_n_{}.pt'.format(n)))
    model.eval()
    x = x.to(device)
    return model.decoder(x).cpu()


def exp3(base_dir, device, x, n):
    model = CNNVAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment3/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    return model.decoder(x).cpu()


def exp4(base_dir, device, x, n):
    model = IWAE(n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment4/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    h2 = torch.randn(10, 100).to(device)
    _, _, p = model.decoder(h2, x)
    return p


def exp6(base_dir, device, x, n):
    noise_sp = .05
    s_p = 'salt_and_pepper'
    model = DVAE(500, n, noise_sp, s_p).to(device).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment6/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    return model.decoder(x).cpu()


def exp7(base_dir, device, x, n):
    model = CVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment7/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()

    c = idx2onehot(torch.arange(0, 10), n=10)
    cc = torch.tensor([c[1].numpy()] * 10)

    x, c = x.to(device), cc.to(device)
    x = x.to(device)

    return model.decoder(x, c)


def exp8(base_dir, device, x, n):
    model = BetaVAE(500, n).to(device)
    model.load_state_dict(torch.load(base_dir / 'experiment8/models/experiment_n_{}.pt'.format(n),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    return model.decoder(x).cpu()


def sample(data, device, base_dir, latent):
    # ---------------------------------------------------------------
    # Get true data and reconstruction
    # ---------------------------------------------------------------
    n = data.shape[0]
    data = data.to(device)

    comparison = torch.cat([
        exp1(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp1_1(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp3(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp4(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp6(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp7(base_dir, device, data, latent).view(n, 1, 28, 28),
        exp8(base_dir, device, data, latent).view(n, 1, 28, 28),
    ])
    sample_dir = Path(base_dir / 'sample')
    sample_dir.mkdir(exist_ok=True, parents=True)

    save_image(comparison.cpu(), sample_dir / 'sample_n_{}.png'.format(latent), nrow=n, padding=1,
               pad_value=1)


def main(n, dir='../data/results'):
    # ---------------------------------------------------------------
    # Setup Configurations
    # ---------------------------------------------------------------
    base_dir = Path(dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X = generate_sample(n, device)

    # ---------------------------------------------------------------
    # Execute and save image
    # ---------------------------------------------------------------
    sample(X, device, base_dir, n)


if __name__ == '__main__':
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        main(n)
