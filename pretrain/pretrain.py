import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path

from models.resnet import ResNet18
from pretrain import utils

# hyperparameters
dataset = 'cifar10'
batch_size=512
val_batch_size=1000
num_epochs=150
learning_rate=2*1e-3
val_ratio = 0.9

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
utils.reproduce()

# define data
data_path = Path(__file__).parent.parent/'datasets'
if dataset == 'cifar10':
    train_data = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True,
                                            transform = transforms.ToTensor())

elif dataset == 'cifar100':
    train_data = torchvision.datasets.CIFAR100(root = data_path, train = True, download = True,
                                            transform = transforms.ToTensor())
else:
    raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

n_train_examples = int(len(train_data) * val_ratio)
n_val_examples = len(train_data) - n_train_examples

train_data, val_data = data.random_split(train_data,
                                    [n_train_examples, n_val_examples])

cifar_stack = torch.stack([img for img, _ in train_data], dim=3)
mean= cifar_stack.view(3,-1).mean(dim=1)
std= cifar_stack.view(3,-1).std(dim=1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data.dataset.transform = transform
val_data.dataset.transform = transform

if dataset == 'cifar10':
    test_data = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True,
                                        transform = transform)
elif dataset == 'cifar100':
    test_data = torchvision.datasets.CIFAR100(root = data_path, train = False, download = True,
                                        transform = transform)
else:
    raise ValueError("Dataset name has to be 'cifar10' or 'cifar100'")

# load data
train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True, **kwargs)
val_loader = data.DataLoader(val_data, batch_size = val_batch_size, shuffle = False, **kwargs)
test_loader = data.DataLoader(test_data, batch_size = val_batch_size, shuffle = False, **kwargs)

    
# load model
model = ResNet18()
model.to(device)

# Define training
def train():
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item() * len(imgs)
        
        train_acc = utils.compute_accuracy(model, train_loader)
        val_acc = utils.compute_accuracy(model, val_loader)
        lr_scheduler.step()
        print(f"Epoch {epoch+1} - Train Loss: {running_loss/len(train_data):.3f} -\
              Train Acc: {train_acc:.2f}% - Validation Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{dataset}_resnet18.pt")


if __name__ == '__main__':
    train()
            


    
    




