import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, h, n):
        super(VAE, self).__init__()
        self.img_size = 28 * 28

        self.h_encoder = nn.Linear(self.img_size, h)
        self.h_mu = nn.Linear(h, n)
        self.h_logvar = nn.Linear(h, n)
        self.h_decoder = nn.Linear(n, h)
        self.output = nn.Linear(h, self.img_size)

    def encoder(self, x):
        h1 = torch.relu(self.h_encoder(x))
        return self.h_mu(h1), self.h_logvar(h1)

    def sample(self, mu, logvar):
        # eq 10
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z ~ p(z|x)

    def decoder(self, z):
        out = torch.relu(self.h_decoder(z))
        return torch.sigmoid(self.output(out))

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, self.img_size))
        z = self.sample(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, x, r, mean, log_var):
        # eq10
        x = x.view(-1, 28 * 28)
        kl = (-0.5 * ((1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)))).sum(1).mean()
        recons = (F.binary_cross_entropy(r, x, reduction='none')).sum(1).mean()
        return kl + recons
