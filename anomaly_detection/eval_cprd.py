import argparse
from pathlib import Path
from collections import defaultdict
import os
import sys

from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.utils import shuffle
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

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


def plot_roc(labels, scores, filename, model_name = '', save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    #plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic rd4ad_{model_name}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


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


def evaluation(model_name, encoder, bn, decoder, data_path, dist = 'L2', size = 32, dataset = 'cifar10', device = 'cpu'):
    plot_dir = Path(__file__).parent / 'CPRD_eval_plots'
    bn.eval()
    encoder.eval()
    decoder.eval()
    gt_list = []
    pr_list = []
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

    with torch.no_grad():
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
    train_maps = train_anomaly_map(encoder, bn, decoder, data_path, dataset = dataset, device = device)
    
    mean = np.mean(train_maps)
    std = np.std(train_maps)
    threshold = mean + 2 * std
    train_score = np.mean(train_maps)
    clean_test_score = np.mean(pr_list[gt_list == 0])
    adv_test_score = np.mean(pr_list[gt_list == 1])
    pred_labels = np.where(pr_list > threshold, 1, 0)
    total_acc = accuracy_score(gt_list, pred_labels)
    # print(f'number of labels: {len(gt_list)}')
    # print(f'number of anomalies: {len(gt_list[gt_list == 1])}')
    detect_rate = np.sum(pred_labels[gt_list == 1] == gt_list[gt_list == 1]) / len(gt_list[gt_list == 1])

   

    auroc = plot_roc(gt_list, pr_list, plot_dir / f'roc_plot_{dataset}_{dist}_{model_name}.png', model_name = model_name, save_plots=True)

    return auroc, detect_rate, total_acc, train_score, clean_test_score, adv_test_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64, help='image size')
    parser.add_argument('--dataset', default = "cifar10", choices = ['cifar10', 'cifar100'])
    parser.add_argument('--dist', default = "L2", choices = ['L2', 'Linf'])
    parser.add_argument('--model', type=str, default = 'resnet18', choices=['resnet152', 'wide_resnet50_2', 'resnet34', 'resnet18', 'resnet50'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_dict = {'resnet152': [resnet152, de_resnet152],
                  'wide_resnet50_2': [wide_resnet50_2, de_wide_resnet50_2],
                  'resnet34': [resnet34, de_resnet34],
                  'resnet18': [resnet18, de_resnet18],
                  'resnet50': [resnet50, de_resnet50]}

    models = model_dict[args.model]
    model_path = Path(__file__).parent / 'CPRD_models'/ f'{args.dataset}_cprd_{args.model}.pth'
    data_path = Path(__file__).parent.parent / 'datasets'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(model_path, map_location='cpu')

    for module in ['bn', 'decoder']:
        temp_dict = {}
        for k, v in state_dict[module].items():
            if 'module' in k:
                k = k.replace('module.', '')
            temp_dict[k] = v
        state_dict[module] = temp_dict

    # for k, v in list(state_dict['bn'].items()):
    #     if 'memory' in k:
    #         state_dict['bn'].pop(k)

    encoder, bn = models[0](pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = models[1](pretrained=False)
    decoder = decoder.to(device)
    decoder.load_state_dict(state_dict['decoder'])
    bn.load_state_dict(state_dict['bn'])
    auroc, detect_rate, total_acc, train_score, clean_test_score, adv_test_score = evaluation(args.model, encoder, bn, decoder, data_path, args.dist, args.size, args.dataset, device)
    print(f'AUROC: {auroc}')
    print(f'Detection Rate: {detect_rate}')
    print(f'Total Accuracy: {total_acc}')
    print(f'train score: {train_score}')
    print(f'clean test score: {clean_test_score}')
    print(f'adv test score: {adv_test_score}')
    result_name = Path(__file__).parent / 'CPRD_eval_plots' / f'{args.dataset}_{args.model}_{args.dist}_result.txt'
    with open(result_name, 'w') as f:
        f.write(f'{args.__dict__}\n')
        f.write(f'AUROC: {auroc}\n')
        f.write(f'Detection Rate: {detect_rate}\n')
        f.write(f'Total Accuracy: {total_acc}\n')
        f.write(f'train score: {train_score}\n')
        f.write(f'clean test score: {clean_test_score}\n')
        f.write(f'adv test score: {adv_test_score}\n')