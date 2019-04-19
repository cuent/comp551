import torch
from src.models.model1 import VAE as BVAE
from src.models.model1_1 import VAE as BVAE_RELU
from src.models.model2 import VAE as GVAE
from src.models.model3 import VAE as CNNVAE
from src.models.model4 import VAE as IWAE
from src.models.model6 import VAE as DVAE
from src.models.model7 import VAE as CVAE
from src.data import MNIST
from src.data import FreyFace
from pathlib import Path
from src.utils import idx2onehot
import numpy as np
import logging


def exp1(base_dir, data, device):
    # Experiment 1
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        model = BVAE(500, n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment1_choose_lr/models/experiment_n_{}.pt'.format(n)))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            y, mu, logvar = model(x)

            loss.append(model.loss_function(x, y, mu, logvar).item())
        logging.info('VAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp1_relu(base_dir, data, device):
    # Experiment 1_1
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        model = BVAE_RELU(500, n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment1_relu/models/experiment_n_{}.pt'.format(n)))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            y, mu, logvar = model(x)

            loss.append(model.loss_function(x, y, mu, logvar).item())
        logging.info('VAE-RELU\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp2(base_dir, data, device):
    # Experiment 2
    ns = [2, 5, 10, 20]
    for n in ns:
        model = GVAE(200, n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment2/models/experiment_n_{}.pt'.format(n)))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            m_enc, logvar_enc, m_dec, logvar_dec = model(x)

            loss.append(model.loss_function(x, m_enc, logvar_enc, m_dec, logvar_dec).item())
        logging.info('VAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp3(base_dir, data, device):
    # Experiment 3
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        model = CNNVAE(n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment3/models/experiment_n_{}.pt'.format(n),
                                         map_location=lambda storage, loc: storage))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            y, mu, logvar = model(x)

            loss.append(model.loss_function(x, y, mu, logvar).item())
        logging.info('CNN VAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp4(base_dir, data, device):
    # Experiment 4
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        model = IWAE(n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment4/models/experiment_n_{}.pt'.format(n),
                                         map_location=lambda storage, loc: storage))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            h1q, h2q, h2p, p = model(x)
            loss.append(model.loss_function(x, h1q, h2q, h2p, p).item())

        logging.info('IWAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp6(base_dir, data, device):
    # Experiment 6
    ns = [3, 5, 10, 20, 200]
    noise_gauss = .25
    noise_sp = .05
    gauss = 'gauss'
    s_p = 'salt_and_pepper'
    for n in ns:
        model = DVAE(500, n, noise_sp, s_p).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment6/models/experiment_n_{}.pt'.format(n),
                                         map_location=lambda storage, loc: storage))
        model.eval()
        loss = []
        for x, _ in data:
            x = x.to(device)
            y, mu, logvar = model(x)

            loss.append(model.loss_function(x, y, mu, logvar).item())
        logging.info('DVAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def exp7(base_dir, data, device):
    # Experiment 7
    ns = [3, 5, 10, 20, 200]
    for n in ns:
        model = CVAE(500, n).to(device)
        model.load_state_dict(torch.load(base_dir / 'experiment7/models/experiment_n_{}.pt'.format(n),
                                         map_location=lambda storage, loc: storage))
        model.eval()
        loss = []
        for x, c in data:
            c = idx2onehot(c, n=10)
            x, c = x.to(device), c.to(device)
            x = x.to(device)
            y, mu, logvar = model(x, c)

            loss.append(model.loss_function(y, x, mu, logvar).item())
        logging.info('CVAE\tn={},elbo={}'.format(n, -np.array(loss).mean()))


def main_discrete(base_dir=Path('../data/results')):
    # ---------------------------------------------------------------
    # Setup Logger
    # ---------------------------------------------------------------
    log_dir = Path(base_dir / 'comparison/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(message)s',
                        handlers=[
                            logging.FileHandler(log_dir / 'comparison'),
                            logging.StreamHandler()
                        ])

    # ---------------------------------------------------------------
    # Common configurations
    # ---------------------------------------------------------------
    torch.manual_seed(590238490)
    data = MNIST.test(64, dir='../data')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    exp1(base_dir, data, device)
    exp1_relu(base_dir, data, device)
    exp3(base_dir, data, device)
    exp4(base_dir, data, device)
    exp6(base_dir, data, device)
    exp7(base_dir, data, device)


def main_continuous(base_dir=Path('../data/results')):
    # ---------------------------------------------------------------
    # Setup Logger
    # ---------------------------------------------------------------
    log_dir = Path(base_dir / 'comparison_continuous/')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s-%(message)s',
                        handlers=[
                            logging.FileHandler(log_dir / 'comparison'),
                            logging.StreamHandler()
                        ])

    # ---------------------------------------------------------------
    # Common configurations
    # ---------------------------------------------------------------
    torch.manual_seed(590238490)
    data = FreyFace.train(64, dir='../data')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    exp2(base_dir, data, device)


if __name__ == '__main__':
    main_discrete()
    main_continuous()
