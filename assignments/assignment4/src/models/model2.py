import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class VAE(nn.Module):
    def __init__(self, h, n):
        super(VAE, self).__init__()
        self.img_size = 28 * 20

        self.h_encoder = nn.Linear(self.img_size, h)
        self.h_mu_encoder = nn.Linear(h, n)
        self.h_logvar_encoder = nn.Linear(h, n)
        self.h_decoder = nn.Linear(n, h)
        self.h_mu_decoder = nn.Linear(h, self.img_size)
        self.h_logvar_decoder = nn.Linear(h, self.img_size)

    def encoder(self, x):
        # flatten data
        x = x.view(-1, self.img_size)
        # encoder
        h_e = torch.tanh(self.h_encoder(x))
        mean = self.h_mu_encoder(h_e)
        log_var = self.h_logvar_encoder(h_e)
        return mean, log_var

    def sample(self, mean, logvar):
        # eq 10
        sd = torch.sqrt(torch.exp(logvar))
        noise = torch.randn_like(sd)
        z = mean + sd * noise  # z ~ q(z|x)
        return z

    def decoder(self, z):
        h = torch.tanh(self.h_decoder(z))
        mean = torch.sigmoid(self.h_mu_decoder(h))
        log_var = torch.tanh(self.h_logvar_decoder(h))
        # log_var = self.h_logvar_decoder(h)
        return mean, log_var

    def forward(self, x):
        mean_enc, logvar_enc = self.encoder(x)
        z = self.sample(mean_enc, logvar_enc)
        mean_dec, logvar_dec = self.decoder(z)
        return mean_enc, logvar_enc, mean_dec, logvar_dec

    def loss_function(self, x, m_enc, logs_enc, m_dec, logs_dec):
        # eq10
        x = x.view(-1, 28 * 20)
        kl = (0.5 * ((1 + logs_enc - torch.pow(m_enc, 2) - torch.exp(logs_enc)))).sum(1)

        # TODO: check loss log p(x|z) for L=1, works only with sigma=1. Diverges when adding true variance.
        log2pi = torch.log(torch.tensor(2 * math.pi))
        # pxz_loss = 0.5 * ((x - m_dec).pow(2) + log2pi).sum(1).mean()

        pxz = -0.5 * (((x - m_dec).pow(2) / torch.exp(logs_dec)) + log2pi + logs_dec).sum(1)
        # mse = -0.5 * (logs_dec + ((x[0] - m_dec[0]).pow(2) * torch.exp(-logs_dec)) + log2pi).sum()
        # mse = -0.5 * (0 + ((x[0] - m_dec[0]).pow(2) / torch.exp(logs_dec))).sum()
        # mse1 = (F.mse_loss(m_dec, x, reduction='none') / 2 * torch.exp(logs_dec)).sum()
        # mse1 += .5 * logs_dec.sum() + x.size()[1] * .5 * torch.log(torch.tensor(2 * math.pi))
        # mse2 = F.mse_loss(m_dec, x, reduction='sum')
        # print(kl.item(), mse.item(), logs_dec.sum().item(), m_dec.sum().item())
        # return kl + ((x[0] - m_dec[0]).pow(2) + log2pi).sum()
        return -(kl + pxz).mean()
