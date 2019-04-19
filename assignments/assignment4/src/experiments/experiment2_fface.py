import torch
from torchvision.utils import save_image
import pickle
from pathlib import Path
import logging
import numpy as np
from torch.nn import utils


def train(model, device, optimizer, epoch, train_loader):
    model.train()

    losses = torch.zeros(len(train_loader))
    num_examples = torch.zeros(len(train_loader))
    for batch, (x, _) in enumerate(train_loader):
        # ---------------------------------------------------------------
        # Training batch
        # ---------------------------------------------------------------
        x = x.to(device)
        optimizer.zero_grad()
        m_enc, log_var_enc, m_dec, log_var_dec = model(x)
        loss = model.loss_function(x, m_enc, log_var_enc, m_dec, log_var_dec)
        loss.backward()
        # print([(p.grad.data.min(), p.grad.data.max()) for p in model.parameters()])
        # utils.clip_grad_value_(model.parameters(), 3)
        # print([(p.grad.data.min(), p.grad.data.max()) for p in model.parameters()])
        optimizer.step()

        # ---------------------------------------------------------------
        # Statistics and stout
        # ---------------------------------------------------------------
        losses[batch] = loss
        num_examples[batch] = (batch + 1) * len(x)
        if batch % 1 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch + 1) * len(x), len(train_loader.dataset),
                       100. * (batch + 1) / len(train_loader),
                loss.item()))

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, losses.mean()))
    return num_examples, losses


def test(model, device, test_loader):
    model.eval()

    losses = torch.zeros(len(test_loader))
    num_examples = torch.zeros(len(test_loader))
    with torch.no_grad():
        for batch, (data, _) in enumerate(test_loader):
            # ---------------------------------------------------------------
            # Testing batch
            # ---------------------------------------------------------------
            data = data.to(device)
            m_enc, log_var_enc, m_dec, log_var_dec = model(data)
            loss = model.loss_function(data, m_enc, log_var_enc, m_dec, log_var_dec).item()

            # ---------------------------------------------------------------
            # Statistics and stout
            # ---------------------------------------------------------------
            losses[batch] = loss
            num_examples[batch] = (batch + 1) * len(data)

    logging.info('====> Test set loss: {:.4f}'.format(losses.mean()))

    return num_examples, losses


def reconstruct(loader, model, device, exp_dir, epoch, latent):
    # ---------------------------------------------------------------
    # Get true data and reconstruction
    # ---------------------------------------------------------------
    data = next(iter(loader))[0]
    data = data.to(device)
    bs = data.shape[0]
    _, _, mean, log_var = model(data)

    # ---------------------------------------------------------------
    # Save images
    # ---------------------------------------------------------------
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n],
                            mean.view(bs, 1, 28, 20)[:n]])
    reconstruct_dir = Path(exp_dir / 'reconstruction/n_{}'.format(latent))
    reconstruct_dir.mkdir(exist_ok=True, parents=True)
    save_image(comparison.cpu(), reconstruct_dir / 'epoch_{}.png'.format(epoch), nrow=n)


def execute(model, optimizer, epochs, train_loader, test_loader, device, n, experiment_dir):
    all_losses_train = []
    all_points_train = []
    all_losses_test = []
    all_points_test = []

    statistics_dir = Path(experiment_dir / "n_{}".format(n))
    statistics_dir.mkdir(exist_ok=True)
    stats = open(statistics_dir / 'loss.txt', "w+")
    stats.write('train' + ',' + 'test' + '\n')

    for epoch in range(1, epochs + 1):
        # ---------------------------------------------------------------
        # Training epoch
        # ---------------------------------------------------------------
        x1, y1 = train(model, device, optimizer, epoch, train_loader)
        all_points_train.append(x1)
        all_losses_train.append(y1)
        # ---------------------------------------------------------------
        # Testing epoch
        # ---------------------------------------------------------------
        x2, y2 = test(model, device, test_loader)
        all_points_test.append(x2)
        all_losses_test.append(y2)
        stats.write(str(y1.mean().item()) + ',' + str(y2.mean().item()) + '\n')
        # ---------------------------------------------------------------
        # Save reconstructions
        # ---------------------------------------------------------------
        reconstruct(test_loader, model, device, experiment_dir, epoch, n)
        # ---------------------------------------------------------------
        # Generate samples
        # ---------------------------------------------------------------
        with torch.no_grad():
            sample = torch.randn(64, n).to(device)
            sample, _ = model.decoder(sample)
            sample = sample.cpu()
            sample_dir = Path(experiment_dir / 'samples/n_{}/'.format(n))
            sample_dir.mkdir(exist_ok=True, parents=True)
            save_image(sample.view(64, 1, 28, 20), sample_dir / 'epoch_{}.png'.format(epoch))

    # ---------------------------------------------------------------
    # Save final model and statistics
    # ---------------------------------------------------------------
    stats.close()
    model_dir = Path(experiment_dir / 'models')
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / 'experiment_n_{}.pt'.format(n))
    _dump_results(all_points_train, all_losses_train, all_points_test, all_losses_test, experiment_dir, n)


def _dump_results(x1, y1, x2, y2, directory, n):
    x1 = np.sum([p.numpy() for p in x1], axis=0)
    y1 = np.mean(np.array([l.detach().numpy() for l in y1]), axis=0)
    x2 = np.sum([p.numpy() for p in x2], axis=0)
    y2 = np.mean(np.array([l.detach().numpy() for l in y2]), axis=0)

    pickle.dump([x1, y1, x2, y2], open(directory / "data_n_{}.pk".format(n), "wb"))
