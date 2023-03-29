import torch

class Cifar(torch.utils.data.Dataset):
    def __init__(self,targets, data, transform = None):
        self.targets = targets
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)
        return img, target