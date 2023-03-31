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
from .utils import *
from .cifar import CIFAR10Data, CIFAR100Data

def train_data(data_path, train_transform, batch_size, dataset = 'cifar10', cls_idx = None):
    # create Training Dataset and Dataloader
    if dataset == 'cifar10':
        train_data = CIFAR10Data(
            root=data_path, train=True, download=True, transform = train_transform)
    elif dataset == 'cifar100':
        train_data = CIFAR100Data(
            root=data_path, train=True, download=True, transform = train_transform)
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    
    if cls_idx is not None:
        targets = train_data.targets
        indices = (np.array(targets) == cls_idx).nonzero()[0]
        train_data = torch.utils.data.Subset(train_data, indices)


    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4,collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)
    return dataloader


def trainCutPaste(epoch, lr, optim_name, size, all_type, batch_size, freeze_resnet, data_path, device = 'cpu', cutpaste = CutPasteNormal, dataset = 'cifar10'):
    def train_model(cls_idx = None):
        # define model
        num_classes = 3 if cutpaste == CutPaste3Way else 2
        model = ProjectionNet(num_classes=num_classes).to(device)
        model.freeze_resnet()
        loss_fn = torch.nn.CrossEntropyLoss()

        if optim_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epoch)
            #scheduler = None
        elif optim_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = None

        # train model
        for e in range(epoch):
            model.train()
            total_loss = 0
            total_num = 0
            if e == freeze_resnet:
                model.unfreeze()
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
                total_loss += loss.item() * len(xc)
                total_num += len(xc)
            print(f'Epoch {e}, Loss {total_loss / total_num}')
            if scheduler is not None:
                scheduler.step(e)
        if cls_idx is None:
            torch.save(model.state_dict(), Path(__file__).parent / 'Cutpaste_models'/f'cutpaste_{dataset}.pth')
        else:
            torch.save(model.state_dict(), Path(__file__).parent / 'Cutpaste_models'/f'cutpaste_{dataset}_{cls_idx}.pth')
    weight_decay = 0.00003
    momentum = 0.9
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
    if all_type:
        num_classes = 10 if dataset == 'cifar10' else 100
        for cls_idx in range(num_classes):
            dataloader = train_data(data_path, train_transform, batch_size, dataset = dataset, cls_idx = cls_idx)
            train_model(cls_idx)
    else:
        dataloader = train_data(data_path, train_transform, batch_size, dataset = dataset)
        train_model()

    

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train models')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--freeze_resnet', default = 30, type=int,
                        help='number of epochs to freeze resnet (default: 30)')
    parser.add_argument('--all_type', default = False, type=str2bool)
    args = parser.parse_args()
    reproduce()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    trainCutPaste(args.epoch, args.lr, args.optim, args.size, args.all_type, args.batch_size, args.freeze_resnet, Path(__file__).parent.parent/'datasets', device = device, dataset = args.dataset)
