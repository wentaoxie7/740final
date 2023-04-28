import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
from cutpaste import CutPasteNormal
from PIL import Image

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']
import torchvision.transforms as transforms
transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_color_new = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                      ])
transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_outliers_loader_new(dataset = 'cifar10', batch_size=8, address="../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth"):
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10
        after_cutpaste_transform.transforms.append(
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]))
    elif dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100
        after_cutpaste_transform.transforms.append(
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]))
    else:
        print("Unexpected Dataset")
        exit()

    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(transforms.Resize((224)))
    train_transform.transforms.append(CutPasteNormal(transform=after_cutpaste_transform))
    coarse = {}

    trainset = ds(root='data', train=True, download=True, transform=train_transform, **coarse)
    #testset0 = ds(root='data', train=False, download=True, transform=transform, **coarse)
    trainset.targets = [1] * len(trainset)
    #testset1 = torch.load(address)['adv_complete']
    #testset1 = testset1.permute(0, 2, 3, 1).detach().cpu().numpy()

    #datas = torch.from_numpy(np.concatenate((testset0.data, testset1), axis=0))
    #targets = [0] * 10000 + [1] * 10000
    #testset = cifar20(targets,datas,transform=transform_new)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              #drop_last=False)
    return train_loader

    #dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    #outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    #return outlier_loader

def get_loaders(dataset, label_class, batch_size):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()
class cifar20(torch.utils.data.Dataset):
    def __init__(self,targets, data, transform = None):
        self.targets = targets
        self.data = data.permute(0,3,1,2)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)
        return img, target

def get_loaders_new(dataset = 'cifar10', batch_size=8, address="../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth"):
    if dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10
    elif dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100
    else:
        print("Unexpected Dataset")
        exit()
    transform = transform_color
    transform_new = transform_color_new
    coarse = {}
    trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
    testset0 = ds(root='data', train=False, download=True, transform=transform, **coarse)
    trainset.targets = [0] * len(trainset)
    testset1 = torch.load(address)['adv_complete']
    testset1 = testset1.permute(0, 2, 3, 1).detach().cpu().numpy()

    datas = torch.from_numpy(np.concatenate((testset0.data, testset1), axis=0))
    targets = [0] * 10000 + [1] * 10000
    testset = cifar20(targets,datas,transform=transform_new)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              drop_last=False)
    return train_loader, test_loader

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


