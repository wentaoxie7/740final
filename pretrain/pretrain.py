'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from pathlib import Path

from models import *
from pretrain.utils import progress_bar
from models.resnet import ResNet18, BasicBlock, ResNet



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net=None, testloader=None, device = None, criterion = None, dataset = None):
    global best_acc
    #the following optional arguments are added for using this function in other script (scope), where such arguments have to be specified.
    #But if calling this function from current script, you don't need to specify them in function call 
    if net is None:
        net = net #from current script(scope)
    if testloader is None:
        testloader = testloader #from current script(scope)
    if device is None:
        device = device #from current script(scope)
    if criterion is None:
        criterion = criterion #from current script(scope)
    if dataset is None:
        dataset = dataset

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # # Save checkpoint for the best accuracy model - (NOT USE HERE BECAUSE WE CANNOT USE TEST DATA TO TELL THE BEST PERFORMANCE MODEL, INSTEAD WE USE THE LAST EPOCH'S MODEL later on)
    # Save checkpoint for all epoch's models (later on, we only use last epoch's model for our final model)
    acc = 100.*correct/total
    # if acc > best_acc:
    #     best_acc = acc
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir(dataset+'checkpoint'):
        os.mkdir(dataset+'checkpoint')
    torch.save(state, './'+dataset+'checkpoint/'+'epoch{}.pt'.format(epoch))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--dataset', '-d', default='cifar10')
    args = parser.parse_args()
    dataset = args.dataset

    if dataset == 'cifar10':
        lr_default = 2*1e-3
    elif dataset == 'cifar100':
        lr_default = 0.1
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

    parser.add_argument('--lr', default=lr_default, type=float, help='default (initial) learning rate ("initial": if using SGD or other to update lr dynamically')
    
    args = parser.parse_args()

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    data_path = Path(__file__).parent.parent/'datasets'
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train_cifar10)
        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test_cifar10)

        batch_size_train = 512
        batch_size_test = 1000

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train_cifar100)
        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test_cifar100)

        batch_size_train = 128
        batch_size_test = 128

    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=4)


    # Model
    print('==> Building model..')
    
    if dataset == 'cifar10':
        net = ResNet18()
    elif dataset == 'cifar100':
        net = ResNet(BasicBlock, [2, 2, 2, 2], 100) #cifar100 has 100 classes
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    # net = VGG('VGG19')
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir(dataset+'checkpoint'), 'Error: no checkpoint directory found!'
    #     if dataset == 'cifar10':
    #         checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     else:
    #         checkpoint = torch.load('./checkpoint/ckpt100.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    if dataset == 'cifar10':
        epochs = 150
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    elif dataset == 'cifar100':
        epochs = 200
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay. In PyTorch, the gamma parameter is used to reduce the learning rate after a certain milestone is reached during training. Specifically, the learning rate is multiplied by gamma after the milestone is reached.

    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")



    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
