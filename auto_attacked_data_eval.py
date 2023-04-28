# To evaluate adversarial data generated from auto attack against the original trained resnet-18 model.
import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

# import sys
# sys.path.insert(0,'..')

from models import *
from models.resnet import ResNet18, BasicBlock, ResNet
import pretrain.pretrain as pretrain
from pretrain.utils import progress_bar


if __name__ == '__main__':
	model_path = Path(__file__).parent/'models'
	adversial_data_path = Path(__file__).parent/'auto_attack_gen_data'
	data_path = Path(__file__).parent/'datasets'
	parser = argparse.ArgumentParser(description='adversarial sample testing')
	parser.add_argument('--dataset', '-d', default='cifar10')
	parser.add_argument('--norm', type=str, default='Linf')
	parser.add_argument('--epsilon', type=float, default=8./255.)
	parser.add_argument('--version', type=str, default='standard')
	args = parser.parse_args()

	dataset = args.dataset
	model_name = dataset+'_model.pt'
	adversial_data_name = '{}_{}_{}_{}_10000_eps_{:.5f}.pth'.format(
				'aa', args.version, args.norm, args.dataset, args.epsilon)

	if dataset == 'cifar10':
	    pass
	elif dataset == 'cifar100':
	    pass
	else:
	    raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

	parser.add_argument('--batch_size', type=int, default=1000) #batch size won't impact test accuracy, only impact training
	parser.add_argument('--adversial_data', type=str, default=os.path.join(adversial_data_path,adversial_data_name))
	parser.add_argument('--model', type=str, default=os.path.join(model_path,model_name))
	parser.add_argument('--data_dir', type=str, default=data_path)

	args = parser.parse_args()


	device = 'cuda' if torch.cuda.is_available() else 'cpu'


	# load model
	if dataset == 'cifar10':
	    net = ResNet18()
	elif dataset == 'cifar100':
	    net = ResNet(BasicBlock, [2, 2, 2, 2], 100) #cifar100 has 100 classes
	else:
	    raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")

	ckpt = torch.load(args.model)
	net.load_state_dict(ckpt['net']) #ckpt is a dict with ket 'net' storing the model weight parameters based on "pretrain.py"
	net = net.to(device)
	criterion = nn.CrossEntropyLoss()


	# load adversarial data
	item = torch.load(args.adversial_data)['adv_complete']
	test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=4)
	
	l = [x for x in test_loader] #adversial images
	x_test = torch.cat(l, 0) #concat to vector
	print(x_test.shape) #torch.Size([10000, 3, 32, 32])
	print(type(x_test)) #<class 'torch.Tensor'>

	# load true label from the same test dataset that was used to do auto-attack
	if dataset == 'cifar10':
		transform_chain = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		item = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
		test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=4)
	elif dataset == 'cifar100':
	    transform_chain = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
	    ])
	    item = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
	    test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=4)
	else:
	    raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")


	l = [y for (x, y) in test_loader] #labels of images
	y_test = torch.cat(l, 0) #concat to vector
	print(y_test.shape) #torch.Size([10000])
	print(type(y_test))	#<class 'torch.Tensor'>


	# Concat adv sample data and true labels to dataloader object
	adv_dataset = data.TensorDataset(x_test, y_test)
	adv_dataset_loader = data.DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


	# test using test function from pretrain.
	epoch_adv = 'adversarial_test_run'+args.norm    
	pretrain.test(epoch_adv, net=net, testloader=adv_dataset_loader, device=device, criterion=criterion, dataset=dataset)
