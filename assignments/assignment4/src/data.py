import torchvision
import torch
import cv2
import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision.datasets.utils import download_url
import torchvision.transforms.functional as FF


class Binaryze(object):
    # https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf
    # http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf
    # Binarization used by Hugo Larochelle in "The Neural Autoregressive Distribution Estimator":
    #   http: // www.dmi.usherb.ca / ~larocheh / mlpython / _modules / datasets / binarized_mnist.html
    def __call__(self, pic):
        img = FF.to_tensor(pic)
        img[img > 0] = .9
        img[img == 0] = 1
        img[img == 0.9] = 0
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MNIST:

    @staticmethod
    def train(bs, dir='../../data'):
        mnist_data = torchvision.datasets.MNIST(dir, download=True, transform=Binaryze())
        data_loader = torch.utils.data.DataLoader(mnist_data,
                                                  batch_size=bs,
                                                  shuffle=True)
        return data_loader

    @staticmethod
    def test(bs, dir='../../data'):
        mnist_data = torchvision.datasets.MNIST(dir, download=True, transform=Binaryze(),
                                                train=False)
        data_loader = torch.utils.data.DataLoader(mnist_data,
                                                  batch_size=bs,
                                                  shuffle=True)
        return data_loader


class FreyFace(data.Dataset):
    filename = "frey_rawface.mat"

    @staticmethod
    def train(bs=100, dir='../../data'):
        frey = FreyFace(dir, download=True, transform=torchvision.transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(frey,
                                                  batch_size=bs,
                                                  shuffle=True)
        return data_loader

    @property
    def folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    def __init__(self, root, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.url = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'

        if download:
            self.download()

        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(self.folder, self.filename))

        self.data = loaded_mat['ff']
        self.data = self.data.T.reshape((-1, 1, 28, 20))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img[0, :, :])

        if self.transform is not None:
            img = self.transform(img)

        return (img, 1)

    def __len__(self):
        return len(self.data)

    def download(self):
        try:
            os.makedirs(self.folder)
        except FileExistsError:
            pass
        if os.path.exists(os.path.join(self.folder, self.filename)):
            return
        download_url(self.url, self.folder, self.filename)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == '__main__':
    bs = 10
    # Frey Faces dataset test
    # for data in FreyFace.train(bs):
    #     for i in range(bs):
    #         img = data[i].numpy().transpose(1, 2, 0)
    #         print(img.shape)
    #         cv2.imshow('', img)
    #         cv2.waitKey()
    # MNIST test
    # for data in MNIST.test(bs):
    #     for i in range(bs):
    #         img = data[0][i].numpy().transpose(1, 2, 0)
    #         lbl = data[1][i]
    #         print(lbl)
    #         cv2.imshow('', img)
    #         cv2.waitKey()
