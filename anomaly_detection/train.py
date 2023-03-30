# train transformaly model from the Transformaly folder on cifar10 and cifar100 for anomaly detection
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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
from .cifar import CIFAR10Data, CIFAR100Data

def train_data(data_path, train_transform, batch_size, dataset = 'cifar10'):
    # create Training Dataset and Dataloader
    if dataset == 'cifar10':
        train_data = CIFAR10Data(
            root=data_path, train=True, download=True, transform = train_transform)
    elif dataset == 'cifar100':
        train_data = CIFAR100Data(
            root=data_path, train=True, download=True, transform = train_transform)
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

    # original_images = [(imgs[0], 0) for imgs, _ in train_data]
    # transformed_images = [(imgs[1], 1) for imgs, _ in train_data]
    # train_data = original_images + transformed_images
    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4,collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)
    return dataloader


def trainCutPaste(epoch, lr, size, batch_size, data_path, device = 'cpu', cutpaste = CutPasteNormal, dataset = 'cifar10'):
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
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpaste(transform = after_cutpaste_transform))
    dataloader = train_data(data_path, train_transform, batch_size, dataset = dataset)
    # define model
    num_classes = 2 
    model = ProjectionNet(num_classes=num_classes).to(device)
    model.freeze_resnet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)

    # train model
    for e in range(epoch):
        model.train()
        for imgs in tqdm(dataloader):
            xs = [x.to(device) for x in imgs]
            xc = torch.cat(xs, dim=0)
            optimizer.zero_grad()
            _, logits = model(xc)
            y = torch.arange(len(xs), device=device)
            y = y.repeat_interleave(xs[0].size(0))
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {e}, Loss {loss.item()}')
        scheduler.step()
    torch.save(model.state_dict(), f'./models/cutpaste_{dataset}.pth')
    
    

if __name__ == '__main__':
    reproduce()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainCutPaste(50, 0.03, 256, 6, Path(__file__).parent.parent/'datasets', device = device)
