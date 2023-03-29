# train transformaly model from the Transformaly folder on cifar10 and cifar100 for anomaly detection
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


from .Cutpaste.model import ProjectionNet
from .Cutpaste.cutpaste import cut_paste_collate_fn, CutPasteNormal, CutPaste3Way
from .utils import reproduce

def train_data(data_path, train_transform, batch_size, dataset = 'cifar10'):
    # create Training Dataset and Dataloader
    if dataset == 'cifar10':
        train_data =torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform = train_transform)
    elif dataset == 'cifar100':
        train_data =torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform = train_transform)
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    
    train_data.data
    original_images = [(imgs[0], 0) for imgs, _ in train_data]
    transformed_images = [(imgs[1], 1) for imgs, _ in train_data]
    train_data = original_images + transformed_images
    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4,# collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)
    return dataloader


def trainCutPaste(epoch, lr, batch_size, data_path, device = 'cpu', cutpaste = CutPasteNormal, dataset = 'cifar10'):
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        after_cutpaste_transform.transforms.append(
            transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010]))
    elif dataset == 'cifar100':
        after_cutpaste_transform.transforms.append(
            transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std = [0.2675, 0.2565, 0.2761]))
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

    train_transform = transforms.Compose([])
    #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    #train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpaste(transform = after_cutpaste_transform))
    dataloader = train_data(data_path, train_transform, batch_size, dataset = dataset)
    # define model
    num_classes = 2 
    model = ProjectionNet(num_classes=num_classes).to(device)
    model.freeze_resnet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train model
    for e in range(epoch):
        model.train()
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)[1]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {e}, Loss {loss.item()}')
    
    

if __name__ == '__main__':
    reproduce()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainCutPaste(200, 0.03, 32, Path(__file__).parent.parent/'datasets', device = device)
