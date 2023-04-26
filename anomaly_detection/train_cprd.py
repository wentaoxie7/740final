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
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score 

from .Cutpaste.cutpaste import cut_paste_collate_fn, CutPasteNormal
from .CPRD.resnet  import wide_resnet50_2, resnet34, resnet18, resnet152, resnet50, ClassifierHead
from .CPRD.de_resnet import de_wide_resnet50_2, de_resnet34, de_resnet18, de_resnet50
from .CPRD.de_resnet import resnet152 as de_resnet152
from .utils import *
from .cifar import CIFAR10Data, CIFAR100Data, Dataset


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


def train_anomaly_map(encoder, bn, decoder, data_path, size = 32,dataset = 'cifar10', device = 'cpu'):
    bn.eval()
    encoder.eval()
    decoder.eval()
    # define transform
    transform = transforms.Compose([])
    if size != 32:
        transform.transforms.append(transforms.Resize((size,size)))

    transform.transforms.append(transforms.CenterCrop(224))
    transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        train_data = CIFAR10Data(
            root= data_path, train = True, download=True, transform = transform)

    elif dataset == 'cifar100':
        transform.transforms.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                            std=[0.2675, 0.2565, 0.2761]))
        train_data = CIFAR100Data(
            root= data_path, train=True, download=True, transform = transform)
    
    dataloader = DataLoader(train_data, batch_size=64,
                                    shuffle=False, num_workers=4, pin_memory= True)
    anomaly_list = []
    with torch.no_grad():
        for img in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs)[0])

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, size)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_list.append(np.sum(anomaly_map, axis = (1, 2)))
            
    return np.concatenate(anomaly_list)


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


    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4,collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5)
    return dataloader


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
        outputs = decoder(bn(inputs)[0])
        
        anomaly_map, _ = cal_anomaly_map(inputs, outputs, size)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        pr_list.append(np.sum(anomaly_map, axis = (1, 2)))
        gt_list.append(label.cpu().numpy())
    pr_list = np.concatenate(pr_list)
    gt_list = np.concatenate(gt_list) 
    auroc = roc_auc_score(gt_list, pr_list)
    clean_test_score = np.mean(pr_list[gt_list == 0])
    adv_test_score = np.mean(pr_list[gt_list == 1])
    print(f'clean test score: {clean_test_score}')
    print(f'adv test score: {adv_test_score}')
    return auroc, pr_list, gt_list


def loss_function(a, b, y):
    #mse_loss = torch.nn.MSELoss()
    cos_sim = torch.nn.CosineSimilarity()
    adv_idx = y == 1
    benign_idx = y == 0
    loss = 0
    weight = [(2)**i for i in range(len(a))]
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1 * mse_loss(a[item], b[item])
        
        loss += weight[item] * torch.mean(-cos_sim(a[item][benign_idx].view(a[item][benign_idx].shape[0],-1),
                                      b[item][benign_idx].view(b[item][benign_idx].shape[0],-1)) 
                           + cos_sim(a[item][adv_idx].view(a[item][adv_idx].shape[0],-1), b[item][adv_idx].view(b[item][adv_idx].shape[0],-1)))
    return loss / np.sum(weight)

def trainCPRD(model, epoch, lr, size,  batch_size, data_path, device = 'cpu', cutpaste = CutPasteNormal, dataset = 'cifar10', parallel = False):
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
    # define model
    num_classes = 2
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
    # define proxy classifier
    rand_input = torch.randn(1, 3, size, size, device=device)
    rand_input = bn(encoder(rand_input))[1]
    classifier_head = ClassifierHead(rand_input.size(1)).to(device)
    classifier_head.train()
    classify_loss_fn = nn.BCEWithLogitsLoss()


    if parallel:
        decoder = torch.nn.DataParallel(decoder)
        bn = torch.nn.DataParallel(bn)
        encoder = torch.nn.DataParallel(encoder)
    
   
    train_loader = train_data(data_path, train_transform, batch_size, dataset)
    test_loader = get_test_loader(size, data_path, dataset)

    # optimizer
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()) + list(classifier_head.parameters()), lr=lr, betas=(0.9,0.999))

    model_dir = Path(__file__).parent / 'CPRD_models'
    model_dir.mkdir(exist_ok=True, parents=True)
    best_acc = 0

   

    # train model
    for e in range(epoch):
        bn.train()
        decoder.train()
        total_loss = 0
        total_num = 0
        for imgs in tqdm(train_loader):
            loss = 0
            xs = [x.to(device) for x in imgs]
            xc = torch.cat(xs, dim=0)
            optimizer.zero_grad()
            encoder_output = encoder(xc)
            bn_output = bn(encoder_output)
            logits = classifier_head(bn_output[1])
            decoder_output = decoder(bn_output[0])

            y = torch.arange(len(xs), device=device)
            y = y.repeat_interleave(xs[0].size(0))
            loss += classify_loss_fn(logits.squeeze(), y.float())
            loss += loss_function(encoder_output, decoder_output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xc)
            total_num += len(xc)
        print(f'Epoch {e + 1}, Loss {total_loss / total_num}')

        train_maps = train_anomaly_map(encoder, bn, decoder, data_path, dataset = dataset, device = device)
        print('Clean training anomaly score is ', np.mean(train_maps))
        auroc, pr_list, gt_list = calAUROC(test_loader, size, encoder, bn, decoder, device)
        threshold = np.mean(train_maps) + 2 * np.std(train_maps)
        pred_labels = np.where(pr_list > threshold, 1, 0)
        total_acc = np.sum(pred_labels == gt_list) / len(gt_list)
        print('Total accuracy is ', total_acc)
        detect_acc = np.sum(pred_labels[gt_list == 1] == gt_list[gt_list == 1]) / len(gt_list[gt_list == 1])
        print('Detection accuracy is ', detect_acc)

        print('AUROC is ', auroc)
        if total_acc > best_acc:
            best_acc = total_acc
            print('Saving best model with acc ', best_acc)
            torch.save({'bn': bn.state_dict(),
                    'decoder': decoder.state_dict(), 'classifier': classifier_head.state_dict()}, model_dir / f'{dataset}_cprd_{model}.pth')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train RD4AD')
    parser.add_argument('--model', type=str, default = 'resnet18', choices=['resnet152', 'wide_resnet50_2', 'resnet34', 'resnet18', 'resnet50'])
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
    trainCPRD(**parse_args.__dict__)

            
   
