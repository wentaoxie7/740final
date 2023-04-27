import argparse
from pathlib import Path
from collections import defaultdict
import os

from sklearn.metrics import roc_curve, auc, accuracy_score
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

def get_train_embeds(model, data_path, transform, device, dataset = 'cifar10', cls_idx = None):
    # train data / train kde
    data = get_train_data(data_path, transform, dataset = dataset)
    if cls_idx is not None:
        targets = data.targets
        indices = (np.array(targets) == cls_idx).nonzero()[0]
        data = torch.utils.data.Subset(data, indices)

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

def eval_model(modelname, data_path, dist = 'L2', dataset = 'cifar10', cls_idx = None, device="cpu", save_plots=False, size=256, model=None, train_embed=None, density=GaussianDensityTorch()):
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

    if cls_idx is not None:
        targets = test_data.targets
        indices = (np.array(targets) == cls_idx).nonzero()[0]
        test_data = torch.utils.data.Subset(test_data, indices)

    target_len = len(test_data)
    adv_path = Path(__file__).parent.parent/'auto_attack_gen_data'/f'aa_standard_{dist}_{dataset}_10000_eps_0.03137.pth'
    if cls_idx is not None:
        adv_data = torch.load(adv_path)['adv_complete']
        indices = (np.array(targets) == cls_idx).nonzero()[0]
        adv_data = torch.utils.data.Subset(adv_data, indices)
        adv_data = Dataset([1] * target_len, adv_data, transforms.Resize((size,size)))
    else:
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
        train_embed = get_train_embeds(model, data_path, test_transform, device = device, dataset = dataset, cls_idx = cls_idx)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    #create eval plot dir
    eval_dir = Path(__file__).parent / 'Cutpaste_eval_plots' 
    if save_plots:
        eval_dir.mkdir(parents=True, exist_ok=True)
        tsne_labels = labels
        tsne_embeds = embeds
        if cls_idx is not None:
            plot_tsne(tsne_labels, tsne_embeds, eval_dir / f"tsne_{dataset}_{cls_idx}_{dist}.png")
        else:
            plot_tsne(tsne_labels, tsne_embeds, eval_dir / f"tsne_{dataset}_{dist}.png")
        
    
    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)
    train_distances = density.predict(train_embed)
    if isinstance(density, GaussianDensityTorch):
        distances = distances.cpu().numpy()
        train_distances = train_distances.cpu().numpy()
    
    mean_distance = np.mean(train_distances)
    stddev_distance = np.std(train_distances)

    threshold = mean_distance + 2 * stddev_distance

    pred_labels = np.where(distances > threshold, 1, 0)
    total_acc = accuracy_score(labels, pred_labels)
    detect_rate = np.sum(pred_labels[labels == 1]) / len(labels[labels == 1])

    train_score = np.mean(train_distances)
    clean_test_score = np.mean(distances[labels == 0])
    adv_test_score = np.mean(distances[labels == 1])
    print(f"train score: {train_score}")
    print(f"clean test score: {clean_test_score}")
    print(f"adv test score: {adv_test_score}")
    
    
    if cls_idx is not None:
        roc_auc = plot_roc(labels, distances, eval_dir / f"roc_plot_{dataset}_{cls_idx}_{dist}.png", modelname=modelname, save_plots=save_plots)
    else:
        roc_auc = plot_roc(labels, distances, eval_dir / f"roc_plot_{dataset}_{dist}.png", modelname=modelname, save_plots=save_plots)
    
    return roc_auc, detect_rate, total_acc, train_score, clean_test_score, adv_test_score
    

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
        plt.title(f'Receiver operating characteristic {os.path.basename(modelname)}')
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

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                    help='save TSNE and roc plots')
    
    parser.add_argument('--dataset', default = "cifar10", choices = ['cifar10', 'cifar100'])
    
    parser.add_argument('--dist', default = "L2", choices = ['L2', 'Linf'])

    parser.add_argument('--all_type', default = False, type=str2bool)

    args = parser.parse_args()

    data_path = Path(__file__).parent.parent / 'datasets'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]
    obj = defaultdict(list)

    if not args.all_type:
        # find models
        model_name = Path(__file__).parent / 'Cutpaste_models' / f'cutpaste_{args.dataset}.pth'

        
        print(f"evaluating {args.dataset}")
        roc_auc, detect_rate, total_acc, train_score, clean_test_score, adv_test_score = eval_model(model_name, data_path, dist = args.dist, dataset = args.dataset, save_plots=args.save_plots, device=device, density=density())
        print(f"AUC: {roc_auc}")
        obj["AUC"].append(roc_auc)
        obj["detection_rate"].append(detect_rate)
        obj["total_acc"].append(total_acc)
        obj["train_score"].append(train_score)
        obj["clean_test_score"].append(clean_test_score)
        obj["adv_test_score"].append(adv_test_score)
        # save pandas dataframe
        eval_dir = Path(__file__).parent / 'Cutpaste_eval_plots'
        eval_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(obj)
        df.to_csv(eval_dir / f"cutpaste_{args.dataset}_{args.dist}_AUC.csv", index=False)

    
    else:
        
        num_classes = 10 if args.dataset == "cifar10" else 100
        for cls_idx in range(num_classes):
            model_name = Path(__file__).parent / 'Cutpaste_models' / f'cutpaste_{args.dataset}_{cls_idx}.pth'
            print(f"evaluating {args.dataset}_{cls_idx}")
            roc_auc, detect_rate, total_acc, train_score, clean_test_score, adv_test_score = eval_model(model_name, data_path, dist = args.dist, dataset = args.dataset, save_plots=args.save_plots, 
                                                         cls_idx = cls_idx, device=device, density=density())
            print(f"AUC: {roc_auc}")
            obj["class_idx"].append(cls_idx)
            obj["AUC"].append(roc_auc)
            obj["detection_rate"].append(detect_rate)
            obj["total_acc"].append(total_acc)
            obj["train_score"].append(train_score)
            obj["clean_test_score"].append(clean_test_score)
            obj["adv_test_score"].append(adv_test_score)
        # save pandas dataframe
        eval_dir = Path(__file__).parent / 'Cutpaste_eval_plots'
        eval_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(obj)
        df.to_csv(eval_dir / f"cutpaste_{args.dataset}_{args.dist}_all_type_AUC.csv", index=False)