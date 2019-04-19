import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class VAE(nn.Module):
    def __init__(self, h, n, noise_factor, noise):
        super(VAE, self).__init__()
        self.img_size = 28 * 28
        self.noise_factor = noise_factor
        self.noise = noise

        # Encoder
        self.enc = nn.Linear(self.img_size, h)
        self.enc_mu = nn.Linear(h, n)
        self.enc_logvar = nn.Linear(h, n)

        # Decoder
        self.dec = nn.Linear(n, h)
        self.out = nn.Linear(h, self.img_size)

        # Initialize weights wit xavier.
        # nn.init.xavier_uniform_(self.enc.weight)
        # nn.init.xavier_uniform_(self.enc_mu.weight)
        # nn.init.xavier_uniform_(self.enc_logvar.weight)
        # nn.init.xavier_uniform_(self.out.weight)

    def encoder(self, x):
        out = torch.relu(self.enc(x))
        mean = self.enc_mu(out)
        log_var = self.enc_logvar(out)

        return mean, log_var

    def sample(self, mean, log_var):
        sd = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(sd)
        z = mean + sd * eps
        return z

    def decoder(self, z):
        out = torch.relu(self.dec(z))
        y = torch.sigmoid(self.out(out))
        return y

    def salt_and_pepper(self, x):
        prob = self.noise_factor
        rnd = torch.rand_like(x)
        noisy = x[:]
        noisy[rnd < prob / 2] = 1.
        noisy[rnd > 1 - prob / 2] = 0.

        # import cv2
        # for i in range(x.size()[0]):
        #     img = x[0].view(1, 28, 28).numpy().transpose(1, 2, 0)
        #     cv2.imshow('', img)
        #     cv2.waitKey()

        # print(noisy.sum().item(), x.sum().item())
        return noisy

    def gauss(self, x):
        x_noise = x + self.noise_factor * torch.rand_like(x)
        x_noise.data.clamp_(0., 1.)
        return x_noise

    def add_noise_to_input(self, x):
        x_clone = x.detach().clone()
        if self.noise == 'salt_and_pepper':
            return self.salt_and_pepper(x_clone)
        elif self.noise == 'gauss':
            self.gauss(x_clone)
        else:
            raise Exception('Provide some noise...')

    def forward(self, x):
        x = x.view(-1, self.img_size)
        x = self.add_noise_to_input(x)
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        y = self.decoder(z)
        return y, mean, log_var

    def loss_function(self, x, p, mean, log_var):
        x = x.view(-1, self.img_size)
        kl = (0.5 * ((1 + log_var - mean.pow(2) - log_var.exp())).sum(1)).mean()
        recons = (-F.binary_cross_entropy(p, x, reduction='none')).sum(1).mean()
        return -(kl + recons)


if __name__ == '__main__':
    model = VAE()
    print(model)
