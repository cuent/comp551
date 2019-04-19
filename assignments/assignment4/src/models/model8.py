# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:54:39 2019

@author: rizvi
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, h, n):
        super(VAE, self).__init__()
        self.img_size = 28 * 28
        self.n = n
        self.h_encoder = nn.Linear(self.img_size, h)
        self.h_mu = nn.Linear(h, self.n)
        self.h_logvar = nn.Linear(h, self.n)
        self.h_decoder = nn.Linear(self.n, h)
        self.output = nn.Linear(h, self.img_size)
        self.beta = 4  # reference: Higgins, 2016

    def encoder(self, x):
        # flatten data
        x = x.view(-1, self.img_size)
        # encoder
        h_e = torch.tanh(self.h_encoder(x))
        mean = self.h_mu(h_e)
        log_var = self.h_logvar(h_e)
        return mean, log_var

    def sample(self, mean, logvar):
        # eq 10
        sd = torch.sqrt(torch.exp(logvar))

        noise = torch.randn_like(sd)  # 0.01 in the paper
        z = mean + sd * noise  # z ~ q(z|x)
        return z

    def decoder(self, z):
        y = torch.sigmoid(self.output(torch.tanh(self.h_decoder(z))))

        return y

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample(mean, logvar)
        y = self.decoder(z)
        return y, mean, logvar

    def loss_function(self, r, x, mean, log_var):
        # eq10
        x = x.view(-1, 28 * 28)
        kl = -0.5 * torch.sum((1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)))
        bce = F.binary_cross_entropy(r, x, reduction='sum')
        return self.beta * kl + bce