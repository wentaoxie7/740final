import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
import sys
from copy import deepcopy
import os
from losses import CompactnessLoss, EWCLoss
from tqdm import tqdm

def save_model(address, model, epoch):
    torch.save(model.state_dict(), address + "/" + str(epoch) + ".pth")
def load_model(address, model, epoch):
    model.load_state_dict(torch.load(address + '/' + str(epoch)+ '.pth'))
    return model
def save_data(address, datas, labels, epoch):
    txt_name = address+"/"+str(epoch)+".txt"
    with open(txt_name,"w") as f:
        for i in range(len(datas)):
            f.writelines(str(datas[i])+ " " + str(labels[i]) + "\n")

def train_model(model, train_loader, outliers_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    bce = torch.nn.BCELoss()
    for epoch in range(args.epochs):
        center = torch.FloatTensor(feature_space).mean(dim=0)
        cle = CompactnessLoss(center.to(device))
        running_loss = run_epoch(model, train_loader, outliers_loader, optimizer, bce, cle, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader,epoch)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, outliers_loader, optimizer, bce, cle, device, ewc, ewc_loss):
    print("start an epoch")
    running_loss = 0.0
    count = 0
    for i, (imgs, _) in enumerate(train_loader):
        count += 1
        if count % 10 == 0:
            print(count)

        imgs = imgs.to(device)

        out_imgs, _ = next(iter(outliers_loader))

        outlier_im = out_imgs.to(device)

        optimizer.zero_grad()

        pred, features = model(imgs)
        outlier_pred, _ = model(outlier_im)

        batch_1 = pred.size()[0]
        batch_2 = outlier_pred.size()[0]

        labels = torch.zeros(size=(batch_1 + batch_2,), device=device)
        labels[batch_1:] = torch.ones(size=(batch_2,))

        loss1 = bce(torch.sigmoid(torch.cat([pred, outlier_pred])), labels)

        loss2 = cle(features)
        if (ewc):
            loss2 += ewc_loss(model)

        loss = loss1 + loss2
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()
    print("end an epoch")

    return running_loss / (i + 1)



def get_score(model, device, train_loader, test_loader, epoch=0):
    print("start getting score")
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)

    distance_0 = []
    distance_1 = []

    for i in range(len(distances)):
        if test_labels[i] == 0:
            distance_0.append(distances[i])
        elif test_labels[i] == 1:
            distance_1.append(distances[i])
    distance_0t = torch.tensor(distance_0).to(device)
    distance_1t = torch.tensor(distance_1).to(device)
    mean_distance = torch.mean(distance_0t)
    stddev_distance = torch.std(distance_0t)
    threshold = mean_distance + 0.5 * stddev_distance

    pred_labels = np.where(torch.tensor(distances).cpu() > threshold.cpu(), 1, 0)
    total_acc = (np.sum((pred_labels == 0) & ((np.array(test_labels)) == 0)) + np.sum(
        pred_labels[(np.array(test_labels)) == 1])) / len(test_labels)
    detect_rate = np.sum(pred_labels[(np.array(test_labels)) == 1]) / (len(test_labels) / 2)
    with open("./result/" + args.save_name + "/distances.txt", "a") as f:
        f.writelines(str(auc) + " " + str(mean_distance.item()) + ' ' + str(stddev_distance.item()) + ' ' +
                     str(threshold.item()) + " " + str(total_acc) + " " + str(detect_rate) +
                     " " + str(sum(distance_0) / len(distance_0)) + " " + str(sum(distance_1) / len(distance_1)) + "\n")
    save_data("./result/" + args.save_name + '/data', distances, test_labels, epoch)

    return auc, train_feature_space
    print("end getting score")
    return auc

def main(args):
    try:
        os.mkdir("./result")
    except:
        pass
    try:
        os.mkdir(r"./result/" + args.save_name)
        os.mkdir(r"./result/" + args.save_name + '/data')
        os.mkdir(r"./result/" + args.save_name + '/model')
    except:
        pass

    print('Dataset: {}, LR: {}'.format(args.dataset, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        for key, value in fisher.items():
            fisher[key] = value.to(device)
        ewc_loss = EWCLoss(frozen_model, fisher)

    #utils.freeze_parameters(model)
    # train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    train_loader, test_loader = utils.get_loaders_new(dataset=args.dataset, batch_size=args.batch_size,
                                                      address=args.adversarial_address)
    outliers_loader = utils.get_outliers_loader_new(dataset=args.dataset, batch_size=args.batch_size,
                                                    address=args.adversarial_address)
    print(type(train_loader.dataset.data))
    print(type(test_loader.dataset.data))

    model = utils.get_resnet_model(resnet_type=args.resnet_type)

    # Change last layer
    model.fc = torch.nn.Linear(args.latent_dim_size, 1)

    model = model.to(device)
    utils.freeze_parameters(model, train_fc=True)

    for name, param in model.named_parameters():
        print(f'{name}: {not param.requires_grad}')
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            print(f'{name}: {layer.in_features}')
            print(f'{name}: {layer.out_features}')

    train_model(model, train_loader, outliers_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    # parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_name', default='abc', help='where to store result')
    parser.add_argument('--adversarial_address',
                        default='../auto_attack_gen_data/aa_standard_Linf_cifar100_10000_eps_0.03137.pth',
                        help='10000 images')
    parser.add_argument('--latent_dim_size', default=2048, type=int)

    args = parser.parse_args()

    main(args)
