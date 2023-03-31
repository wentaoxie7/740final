import torch
import torchvision
from torchvision import transforms

from .Cutpaste.cutpaste import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self,targets, data, transform = None):
        self.targets = targets
        self.data = data
        self.transform = transform
            

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, target

class CIFAR10Data(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img

class CIFAR100Data(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img
    

