import torch.nn as nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, h, n):
        super(VAE, self).__init__()

        self.img_size = 28 * 28

        self.h_encoder = nn.Linear(self.img_size + 10, h)
        self.h_mu = nn.Linear(h, n)
        self.h_logvar = nn.Linear(h, n)
        self.h_decoder = nn.Linear(n + 10, h)
        self.output = nn.Linear(h, self.img_size)

    def encoder(self, x, c):
        # flatten data
        x = x.view(-1, self.img_size)
        x = torch.cat([x, c], 1)
        # encoder
        h_e = torch.tanh(self.h_encoder(x))
        mean = self.h_mu(h_e)
        log_var = self.h_logvar(h_e)
        return mean, log_var

    def sample(self, mean, logvar):
        # eq 10
        sd = torch.sqrt(torch.exp(logvar))
        noise = torch.randn_like(sd)  # N(0, sigma^2)
        z = mean + sd * noise  # z ~ q(z|x) 
        return z

    def decoder(self, z, c):
        z = torch.cat([z, c], dim=1)

        y = torch.sigmoid(self.output(torch.tanh(self.h_decoder(z))))

        return y

    def forward(self, x, c):
        mean, logvar = self.encoder(x, c)
        z = self.sample(mean, logvar)
        y = self.decoder(z, c)
        return y, mean, logvar

    def loss_function(self, r, x, mean, log_var):
        # eq10
        x = x.view(-1, 28 * 28)
        kl = -0.5 * torch.sum((1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)))
        bce = F.binary_cross_entropy(r, x, reduction='sum')
        return kl + bce
