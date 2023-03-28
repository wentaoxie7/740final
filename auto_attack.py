import os
import argparse
from pathlib import Path
import warnings
import sys

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

# import sys
# sys.path.insert(0,'..')

from models import *
from models.resnet import ResNet18, BasicBlock, ResNet

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent/'datasets'
    model_path = Path(__file__).parent.parent/'models'
    output_path = Path(__file__).parent.parent/'auto_attack_gen_data'
    log_path = Path(__file__).parent.parent/'logs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='cifar10')
    parser.add_argument('--norm', type=str, default='Linf')
    args = parser.parse_args()
    dataset = args.dataset
    model_name = dataset+'_model.pt'

    if dataset == 'cifar10':
        pass
    elif dataset == 'cifar100':
        pass
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    parser.add_argument('--data_dir', type=str, default=data_path)
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default=os.path.join(model_path,model_name))
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default=output_path)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--log_path', type=str, default=os.path.join(log_path,'auto_attack_log_file.txt'))
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    
    args = parser.parse_args()

    # load model
    if dataset == 'cifar10':
        model = ResNet18()
    elif dataset == 'cifar100':
        model = ResNet(BasicBlock, [2, 2, 2, 2], 100) #cifar100 has 100 classes
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt['net']) #ckpt is a dict with ket 'net' storing the model weight parameters based on "pretrain.py"
    model.cuda()
    model.eval()

    # load data

    if dataset == 'cifar10':
        transform_chain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        item = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=4)
    elif dataset == 'cifar100':
        transform_chain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        item = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=4)
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

    
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader] #images
    x_test = torch.cat(l, 0) #concat to vector
    l = [y for (x, y) in test_loader] #labels of images
    y_test = torch.cat(l, 0) #concat to vector
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                bs=args.batch_size, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_{}_{}_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, args.norm, args.dataset, adv_complete.shape[0], args.epsilon))
            #note: the saved .pth file contains adv_complete as the value for key 'adv_complete'.
            # adv_complete object is the adversial images generated. It should have the same format as the input images to this script.
            # In this case, it should be a torch.Tensor object with shape as torch.Size([10000, 3, 32, 32])
        else:
            pass
            # # individual version, each attack is run on all test points
            # adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
            #     y_test[:args.n_ex], bs=args.batch_size)
            
            # torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
            #     args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))