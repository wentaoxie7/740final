import torch
from torchvision.datasets

# Load the dataset from the .pth file
#dataset = torch.load('../../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth')['adv_complete']
path = '/home/wentao/ece740/auto_attack_gen_data/aa_standard_Linf_cifar10_10000_eps_0.03137.pth'
dataset = torch.load(path)['adv_complete']
dataset2 = torch.datasets.CIFAR10
# Use the dataset object
# For example, get the number of samples in the dataset
print(dataset.shape)
#print(f'The dataset contains {num_samples} samples.')
