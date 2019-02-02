__author__ = 'Bingqing Wei'
import torch, torchvision
from torch.utils.data import Dataset
import numpy as np

def get_c10():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    return trainset, testset

def get_mnist():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]
         #torchvision.transforms.Normalize((0.5, 0.5), (0.5, 0.5))]
    )

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=True,
                                         download=True, transform=transform)
    return trainset, testset

class MyDataset(Dataset):
    def __init__(self, batchsize, trainset, testset):
        self.trainset, self.testset = trainset, testset
        self.mode = 'train'
        self.batchsize = batchsize

    def set_mode(self, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return int(len(self.trainset) / self.batchsize)
        else: return int(len(self.testset) / self.batchsize)

    @property
    def dataset(self):
        if self.mode == 'train':
            return self.trainset
        else: return self.testset

    def __transform_x__(self, img):
        img = img.numpy()
        return np.reshape(img, np.prod(img.shape))

    def __transform_y__(self, y):
        new_y = np.zeros((10,))
        new_y[y] = 1
        return new_y

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batchsize):
            x, y = self.dataset[(idx * self.batchsize + i) % self.__len__()]
            batch_x.append(self.__transform_x__(x))
            batch_y.append(self.__transform_y__(y))
        return np.array(batch_x), np.array(batch_y)

