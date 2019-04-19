import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class VAE(nn.Module):
    def __init__(self, n=50):
        super(VAE, self).__init__()
        self.img_size = 28 * 28

        # ---------------------------------------------------------------
        # Architecture see Appendix C of IWAE: https://arxiv.org/pdf/1509.00519.pdf
        # ---------------------------------------------------------------

        # Encoder
        self.h1_enc = nn.Sequential(nn.Linear(self.img_size, 200),
                                    nn.Tanh(),
                                    nn.Linear(200, 200),
                                    nn.Tanh())
        self.h1_mu_enc = nn.Linear(200, 100)
        self.h1_log_var_enc = nn.Linear(200, 100)

        self.h2_enc = nn.Sequential(nn.Linear(100, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 100),
                                    nn.Tanh())
        self.h2_mu_enc = nn.Linear(100, n)
        self.h2_log_var_enc = nn.Linear(100, n)

        # Decoder
        self.h2_dec = nn.Sequential(nn.Linear(n, 100),
                                    nn.Tanh(),
                                    nn.Linear(100, 100),
                                    nn.Tanh())
        self.h2_mu_dec = nn.Linear(100, 100)
        self.h2_log_var_dec = nn.Linear(100, 100)

        self.h1_decoder = nn.Sequential(nn.Linear(100, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, self.img_size),
                                        nn.Sigmoid())

    def encoder(self, x):
        out1 = self.h1_enc(x)
        mean1 = self.h1_mu_enc(out1)
        var1 = self.h1_log_var_enc(out1).exp()
        h1 = self.sample(mean1, var1)

        out2 = self.h2_enc(h1)
        mean2 = self.h2_mu_enc(out2)
        var2 = self.h2_log_var_enc(out2).exp()
        h2 = self.sample(mean2, var2)

        return h1, mean1, var1, \
               h2, mean2, var2

    def sample(self, mean, var):
        sd = torch.sqrt(var)
        eps = torch.randn_like(sd)
        h = mean + sd * eps  # h ~ q(h|x)
        return h

    def decoder(self, h1, h2):
        out = self.h2_dec(h2)
        mean = self.h2_mu_dec(out)
        var = self.h2_log_var_dec(out).exp()

        p = self.h1_decoder(h1)
        return mean, var, p

    def forward(self, x):
        # flatten data
        k = x.size()[0]
        x = x.view(k, -1, self.img_size)

        # VAE
        h1, uq1, varq1, h2, uq2, varq2 = self.encoder(x)

        up1, varp1, p = self.decoder(h1, h2)

        return (h1, uq1, varq1), (h2, uq2, varq2), (up1, varp1), p

    def loss_function(self, x, q1, q2, p1, p):
        k = x.size()[0]
        x = x.view(k, -1, self.img_size)

        log2pi = torch.tensor(2 * math.pi).log()

        h1, uq1, varq1 = q1
        h2, uq2, varq2 = q2
        up1, varp1 = p1

        logqh1x = -0.5 * (((h1 - uq1).pow(2) / varq1) + varq1.log() + log2pi).sum(-1)  # log q(h1|x)
        logqh2h1 = -0.5 * (((h2 - uq2).pow(2) / varq2) + varq2.log() + log2pi).sum(-1)  # log q(h2|h1)
        logph1h2 = -0.5 * (((h1 - up1).pow(2) / varp1) + varp1.log() + log2pi).sum(-1)  # log p(h1|h2)
        # ---------------------------------------------------------------
        # Computing reconstruccion error log p(x|h1) = log Bernoulli(x;p) = -BCE.
        #   a) (1 - p).log() = log 0 = nan using plain Pytorch.
        #   b) pytorch BCE implementation gives -0.0000.
        # Using BCE to alleviate the problem nan problems.
        # ---------------------------------------------------------------

        # logpxh1 = (x * p.log() + (1 - x) * (1 - p).log()).sum(-1)  # log p(x|h1)
        #
        logpxh1 = (-F.binary_cross_entropy(p, x, reduction='none')).sum(-1)  # log p(x|h1)
        logph1 = -0.5 * (log2pi + h1.pow(2)).sum(-1)  # log p(h1) -----
        logph2 = -0.5 * (log2pi + h2.pow(2)).sum(-1)  # log p(h2)

        # log importance weights
        logw = logpxh1 + logph1h2 + logph1 + logph2 - logqh2h1 - logqh1x  # eq. 15
        logw = logw - torch.max(logw, 0)[0]  # normalize to avoid overflow when taking the exp

        if len((1 - p)[(1 - p) == 0]) > 0:
            # print('zero-probability found, replacing log 0')
            pass

        # normalized importance weights
        w_hat = logw.exp()
        # w_hat = w_hat.detach()
        w_hat = w_hat / w_hat.sum(0)
        w_hat = w_hat.clone().detach()

        # print(logpxh1 + logph1h2 + logph2 - logqh2h1 - logqh1x)
        elbo = (w_hat * (logpxh1 + logph1h2 + logph1 + logph2 - logqh2h1 - logqh1x)).sum(0).mean()

        return -elbo


if __name__ == '__main__':
    model = VAE()
    print(model)
