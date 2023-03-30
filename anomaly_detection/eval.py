import argparse
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.utils import shuffle

from .Cutpaste.cutpaste import CutPaste
from .Cutpaste.model import ProjectionNet
from .Cutpaste.cutpaste import CutPaste, cut_paste_collate_fn
from .Cutpaste.density import GaussianDensitySklearn, GaussianDensityTorch
from .utils import str2bool
from .cifar import *

test_data_eval = None
test_transform = None
cached_type = None

def get_train_data(data_path, transform, dataset = 'cifar10'):
    # create Training Dataset and Dataloader
    if dataset == 'cifar10':
        train_data =torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform = transform)
    elif dataset == 'cifar100':
        train_data =torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform = transform)
    else:
        raise ValueError("Dataset name has to be 'cifar10' or 'cifar100' ")
    
    return train_data

def get_train_embeds(model, data_path, transform, device, dataset = 'cifar10'):
    # train data / train kde
    data = get_train_data(data_path, transform, dataset = dataset)

    dataloader_train = DataLoader(data, batch_size=64,
                            shuffle=False, num_workers=4)
    train_embed = []
    model.to(device)
    with torch.no_grad():
        for x, _ in dataloader_train:
            embed, logit = model(x.to(device))
            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_model(modelname, data_path, dist = 'L2', dataset = 'cifar10', device="cpu", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, density=GaussianDensityTorch()):
    # create test dataset
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size,size)))
    test_transform.transforms.append(transforms.ToTensor())
    if dataset == 'cifar10':
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        test_data = CIFAR10Data(
        root= data_path, train=False, download=True, transform = test_transform)
    elif dataset == 'cifar100':
        test_transform.transforms.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                            std=[0.2675, 0.2565, 0.2761]))
        test_data = CIFAR100Data(
        root= data_path, train=False, download=True, transform = test_transform)
    
    target_len = len(test_data)
    adv_path = Path(__file__).parent.parent/'auto_attack_gen_data'/f'aa_standard_{dist}_{dataset}_10000_eps_0.03137.pth'
    adv_data = Dataset([1] * target_len, torch.load(adv_path)['adv_complete'], transforms.Resize((size,size)))
    test_data = Dataset([0] * target_len, test_data)
    data = torch.utils.data.ConcatDataset([test_data, adv_data])
    dataloader_test = DataLoader(data, batch_size=64,
                                    shuffle=False, num_workers=4)

    # create model
    if model is None:
        print(f"loading model {modelname}")
        # head_layers = [512]*head_layer+[128]
        # print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    #get embeddings for test data
    labels = []
    embeds = []
    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x.to(device))

            # save 
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, data_path, test_transform, device = device, dataset = dataset)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    #create eval plot dir
    if save_plots:
        eval_dir = Path("eval") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # plot tsne
        # also show some of the training data
        show_training_data = False
        if show_training_data:
            #augmentation setting
            # TODO: do all of this in a separate function that we can call in training and evaluation.
            #       very ugly to just copy the code lol
            min_scale = 0.5

            # create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))

            train_transform = transforms.Compose([])
            #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            #train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())

            train_data = MVTecAT("Data", defect_type, transform=train_transform, size=size)
            dataloader_train = DataLoader(train_data, batch_size=32,
                        shuffle=True, num_workers=8, collate_fn=cut_paste_collate_fn,
                        persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with torch.no_grad():
                for x1, x2 in dataloader_train:
                    x = torch.cat([x1,x2], axis=0)
                    embed, logit = model(x.to(device))

                    # generate labels:
                    y = torch.tensor([0, 1])
                    y = y.repeat_interleave(x1.size(0))

                    # save 
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = torch.cat(train_labels)
            train_embeds = torch.cat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = torch.cat([labels, train_labels + 2])
            tsne_embeds = torch.cat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")
    
    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)
    #TODO: set threshold on mahalanobis distances and use "real" probabilities

    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    

    return roc_auc
    

def plot_roc(labels, scores, filename, modelname="", save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

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
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc

def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:,0], tsne_results[:,1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=False, type=str2bool,
                    help='save TSNE and roc plots')
    
    parser.add_argument('--dataset', default = "cifar10", choices = ['cifar10', 'cifar100'])
    

    args = parser.parse_args()

    data_path = Path(__file__).parent.parent / 'datasets'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # find models
    model_name = Path(__file__).parent.parent / 'models' / f'cutpaste_{args.dataset}.pth'

    obj = defaultdict(list)
    print(f"evaluating {args.dataset}")
    roc_auc = eval_model(model_name, data_path, save_plots=args.save_plots, device=device, density=density())
    print(f"AUC: {roc_auc}")
    
    # # save pandas dataframe
    # eval_dir = Path("eval") / args.model_dir
    # eval_dir.mkdir(parents=True, exist_ok=True)
    # df = pd.DataFrame(obj)
    # df.to_csv(str(eval_dir) + "_perf.csv")