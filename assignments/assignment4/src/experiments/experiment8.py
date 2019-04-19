import torch
from torchvision.utils import save_image
import os
import numpy as np
import logging
from pathlib import Path


class MNIST_vae():
    def __init__(self, model, optimizer, learning_rate, batch_size, n, train_loader, test_loader):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n = n
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epoch):

        self.model.train()
        train_loss = 0
        losses = torch.zeros(len(self.train_loader))
        data_points = torch.zeros(len(self.train_loader))

        for batch, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            y, m, logvar = self.model(x)
            #           TODO: mse, kl
            loss = self.model.loss_function(y, x, m, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            losses[batch] = loss / len(x)
            data_points[batch] = epoch * (batch + 1) * len(x)

            if batch % 100 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch * len(x), len(self.train_loader.dataset),
                           100. * batch / (len(self.train_loader)),
                           loss.item() / len(x)))

        logging.info('====> Epoch: {} Average Loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

        return data_points, losses

    def test(self, epoch):

        recon_dir = 'data/reconstructed_images/' + str(self.n)
        try:
            os.makedirs(recon_dir)
        except:
            pass

        self.model.eval()
        test_loss = 0
        losses = torch.zeros(len(self.test_loader))
        data_points = torch.zeros(len(self.test_loader))
        with torch.no_grad():
            for batch, (x, _) in enumerate(self.test_loader):
                x = x.to(self.device)
                y, mu, logvar = self.model(x)

                loss = self.model.loss_function(y, x, mu, logvar)
                test_loss += loss.item()
                losses[batch] = loss / len(x)
                data_points[batch] = epoch * (batch + 1) * len(x)

                if batch == 0:
                    n = min(x.size(0), 8)
                    comparison = torch.cat([x[:n], y.view(self.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison, os.path.join(recon_dir, 'reconst_{}.png'.format(epoch)))

        #                TODO:  save compared images ==> DONE!

        test_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_loss))

        return data_points, losses

    def execute(self, num_epochs):

        sample_dir = 'data/sample_images/' + str(self.n)
        try:
            os.makedirs(sample_dir)
        except:
            pass

        #
        #        seq_train_loss = torch.zeros(len(self.train_loader))
        #        seq_test_loss = torch.zeros(len(self.test_loader))
        seq_train_loss = []
        seq_test_loss = []
        #        n = dim of latent space
        for epoch in range(1, num_epochs + 1):
            train_points, train_losses = self.train(epoch)
            seq_train_loss.append(train_losses)
            test_points, test_losses = self.test(epoch)
            seq_test_loss.append(test_losses)
            with torch.no_grad():
                sample = torch.randn(64, self.n).to(self.device)
                sample = self.model.decoder(sample)
                save_image(sample.view(64, 1, 28, 28), os.path.join(sample_dir, 'sample_{}.png'.format(epoch)))

        model_dir = Path('saved_models')
        model_dir.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / 'experiment_n_{}.pt'.format(self.n))

        seq_train_loss = np.array([l.detach().numpy() for l in seq_train_loss])
        seq_test_loss = np.array([l.detach().numpy() for l in seq_test_loss])

        return train_points, train_losses, test_points, test_losses, seq_train_loss, seq_test_loss
