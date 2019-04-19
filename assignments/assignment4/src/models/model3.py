import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


class VAE(nn.Module):
    def __init__(self, n):
        super(VAE, self).__init__()
        self.img_size = 28 * 28

        # Encoder
        self.enc_cnn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=3),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.ReLU(),
        )
        # self.enc_fc1 = nn.Linear(8 * 8 * 64, 800)
        self.enc_mu = nn.Linear(8 * 8 * 64, n)
        self.enc_log_var = nn.Linear(8 * 8 * 64, n)

        # Decoder
        self.dec_fc1 = nn.Linear(n, 8 * 8 * 64)
        # self.dec_fc2 = nn.Linear(800, 7 * 7 * 64)
        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1), # 32x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.MaxUnpool2d(kernel_size=5, stride=2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # CenterCrop(28, 28),
            nn.Sigmoid()
        )

    def encoder(self, x):
        out = self.enc_cnn(x)
        out = out.view(out.size(0), -1)
        # out = self.enc_fc1(out)
        mean = self.enc_mu(out)
        log_var = self.enc_log_var(out)

        return mean, log_var

    def sample(self, mean, log_var):
        # eq 10
        sd = torch.sqrt(torch.exp(log_var))
        noise = torch.randn_like(sd)
        z = mean + sd * noise  # z ~ q(z|x)
        return z

    def decoder(self, z):
        out = self.dec_fc1(z)
        # out = F.relu(self.dec_fc2(out))
        out = out.view(out.size(0), 64, 8, 8)
        y = self.dec_cnn(out)
        return y

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        y = self.decoder(z)
        return y, mean, log_var

    def loss_function(self, x, p, mean, log_var):
        # eq10
        bs = x.size()[0]
        kl = (-0.5 * ((1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)))).sum(1).mean()
        bce = F.binary_cross_entropy(p, x, reduction='sum') / bs
        return kl + bce


if __name__ == '__main__':
    model = VAE(10)
    print(model)
