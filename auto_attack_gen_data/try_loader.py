import torch
import torchvision
import numpy as np
from torchvision import transforms
transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_color_new = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                          transforms.ToTensor()
                                      ])
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
dataset = torch.load("./aa_standard_L2_cifar10_10000_eps_0.03137.pth")['adv_complete']
print(dataset.shape)
dataset = dataset.permute(0,2,3,1).detach().cpu().numpy()

print(dataset.shape)
ds = torchvision.datasets.CIFAR10
transform = transform_color
coarse = {}
trainset = ds(root='data', train=True, download=True, transform=transform_color, **coarse)
testset = ds(root='data', train=False, download=True, transform=transform_color, **coarse)
print(testset.data.shape)
testset.data = torch.from_numpy(np.concatenate((testset.data, dataset), axis=0))
testset.targets= [0]*10000 + [1]*10000
data = cifar20(testset.targets, testset.data, transform = transform_color_new)
print(data.__getitem__(9999))
#print(testset.targets)
#print(testset)