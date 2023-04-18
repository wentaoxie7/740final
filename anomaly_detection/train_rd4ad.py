from pathlib import Path
import os
import argparse

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score  

from .RD4AD.resnet  import wide_resnet50_2, resnet34, resnet18, resnet152, resnet50
from .RD4AD.de_resnet import de_wide_resnet50_2, de_resnet34, de_resnet18, de_resnet50
from .RD4AD.de_resnet import resnet152 as de_resnet152
from .cifar import CIFAR10Data, CIFAR100Data, Dataset
from .utils import reproduce, str2bool

def cal_anomaly_map(fs_list, ft_list, out_size=32):
    anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs = F.normalize(fs, p=2)
        ft = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0, :, :].detach().cpu()
        a_map_list.append(a_map)
        anomaly_map += a_map.cpu().numpy()
    return anomaly_map, a_map_list

def get_test_loader(size, data_path, dataset = 'cifar10', dist = 'L2'):
    # define transform
    transform = transforms.Compose([])
    adv_transform = transforms.Compose([])
    if size != 32:
        transform.transforms.append(transforms.Resize((size,size)))

    transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        adv_transform.transforms.append(transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225]))

        test_data = CIFAR10Data(
            root= data_path, train=False, download=True, transform = transform)

    elif dataset == 'cifar100':
        transform.transforms.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761]))
        adv_transform.transforms.append(transforms.Normalize(mean = [-0.5071/0.2675, -0.4867/0.2565, -0.4408/0.2761], std = [1/0.2675, 1/0.2565, 1/0.2761]))

        test_data = CIFAR100Data(
            root= data_path, train=False, download=True, transform = transform)
    
    adv_transform.transforms.append(transforms.ToPILImage())
    if size != 32:
        adv_transform.transforms.append(transforms.Resize((size,size)))
   
    adv_transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        adv_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
    elif dataset == 'cifar100':
        adv_transform.transforms.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                            std=[0.2675, 0.2565, 0.2761]))
    target_len = len(test_data)
    adv_path = Path(__file__).parent.parent/'auto_attack_gen_data'/f'aa_standard_{dist}_{dataset}_10000_eps_0.03137.pth'

    adv_data = Dataset([1] * target_len, torch.load(adv_path)['adv_complete'], 
                       transform = adv_transform)
    test_data = Dataset([0] * target_len, test_data)
    data = torch.utils.data.ConcatDataset([test_data, adv_data])
    dataloader = DataLoader(data, batch_size=64,
                                    shuffle=False, num_workers=4, pin_memory = True)

    return dataloader


@torch.no_grad()
def calAUROC(dataloader, size, encoder, bn, decoder, device):
    bn.eval()
    decoder.eval()
    gt_list = []
    pr_list = []
    for img, label in dataloader:
        img = img.to(device)
        inputs = encoder(img)
        outputs = decoder(bn(inputs))
        
        anomaly_map, _ = cal_anomaly_map(inputs, outputs, size)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        pr_list.append(np.sum(anomaly_map, axis = (1, 2)))
        gt_list.append(label.cpu().numpy())
    pr_list = np.concatenate(pr_list)
    gt_list = np.concatenate(gt_list) 
    auroc = roc_auc_score(gt_list, pr_list)
    return auroc

def loss_function(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1 * mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss



def trainRD4AD(model, epoch, lr, batch_size, size, data_path, device = 'cpu', dataset = 'cifar10', parallel = False):
    # define transform
    transform = transforms.Compose([])
    if size != 32:
        transform.transforms.append(transforms.Resize((size,size)))
   
    transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        train_data = CIFAR10Data(
            root= data_path, train=True, download=True, transform = transform)

    elif dataset == 'cifar100':
        transform.transforms.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                            std=[0.2675, 0.2565, 0.2761]))
        train_data = CIFAR100Data(
            root= data_path, train=True, download=True, transform = transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True, prefetch_factor=4)

    # define model
    model_dict = {'resnet152': [resnet152, de_resnet152],
                  'wide_resnet50_2': [wide_resnet50_2, de_wide_resnet50_2],
                  'resnet34': [resnet34, de_resnet34],
                  'resnet18': [resnet18, de_resnet18],
                  'resnet50': [resnet50, de_resnet50]
                  }
    
    models = model_dict[model]
    encoder, bn = models[0](pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = models[1](pretrained=False)
    decoder = decoder.to(device)
    encoder.eval()
    bn.train()
    decoder.train()
    if parallel:
        decoder = torch.nn.DataParallel(decoder)
        bn = torch.nn.DataParallel(bn)
        encoder = torch.nn.DataParallel(encoder)

    # optimizer
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=lr, betas=(0.9,0.999))
    # optimizer = torch.optim.SGD(list(decoder.parameters())+list(bn.parameters()), lr=lr, weight_decay=0.00005, momentum=0.9)

    model_dir = Path(__file__).parent / 'RD4AD_models'
    model_dir.mkdir(exist_ok=True, parents=True)
    best_auroc = 0.8
    # test loader
    test_loader = get_test_loader(size, data_path, dataset)

    # train
    for e in range(epoch):
        bn.train()
        decoder.train()
        loss_list = []
        for img in tqdm(train_dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_function(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        
        print('epoch [{}/{}], loss:{:.4f}'.format(e + 1, epoch, np.mean(loss_list)))
        auroc = calAUROC(test_loader, size, encoder, bn, decoder, device)
        if auroc > best_auroc:
            best_auroc = auroc
            print(f'New best AUROC is {best_auroc}')
            torch.save({'bn': bn.state_dict(),
                    'decoder': decoder.state_dict()}, model_dir / f'{dataset}_rd4ad_{model}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train RD4AD')
    parser.add_argument('--model', type=str, default = 'resnet34', choices=['resnet152', 'wide_resnet50_2', 'resnet34', 'resnet18', 'resnet50'])
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default= 64, help='batch size')
    parser.add_argument('--size', type=int, default= 64, help='image size')
    parser.add_argument('--data_path', type=str, default= './datasets', help='data path')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--parallel', type= str2bool, default=False, help='use parallel')
    parse_args = parser.parse_args()
    reproduce()
    parse_args.device = parse_args.device if torch.cuda.is_available() else 'cpu'
    trainRD4AD(**parse_args.__dict__)

