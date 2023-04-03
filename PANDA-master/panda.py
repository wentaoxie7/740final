import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import os
import sys
def save_model(address, model, epoch):
    torch.save(model.state_dict(), address + "/" + str(epoch) + ".pth")
def load_model(address, model, epoch):
    # model = resnet50(weights=torch.load(path + '/' + name + '.pth'))
    model.load_state_dict(torch.load(address + '/' + str(epoch)+ '.pth'))
    return model
def save_data(address, datas, labels, epoch):
    txt_name = address+"/"+str(epoch)+".txt"
    with open(txt_name,"w") as f:
        for i in range(len(datas)):
            f.writelines(str(datas[i])+ " " + str(labels[i]) + "\n")

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    save_model("./result/"+args.save_name+'/model', model, 0)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        save_model("./result/"+args.save_name+'/model', model, epoch)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader, epoch)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader, epoch=0):
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
    distance_0t = torch.tensor(distance_0)
    distance_1t = torch.tensor(distance_1)
    mean_distance = torch.mean(distance_0t)
    stddev_distance = torch.std(distance_0t)
    threshold = mean_distance + 2 * stddev_distance

    pred_labels = np.where(torch.tensor(distances) > threshold, 1, 0)
    total_acc = (np.sum((pred_labels==0) & ((np.array(test_labels))==0))+ np.sum(pred_labels[(np.array(test_labels))==1]))/len(test_labels)
    detect_rate = np.sum(pred_labels[(np.array(test_labels))==1])/ (len(test_labels)/2)
    with open("./result/"+args.save_name+"/distances.txt","a") as f:
        f.writelines(str(auc)+ " " + str(mean_distance.numpy) + ' ' + str(stddev_distance.numpy) + ' '+
                     str(threshold.numpy) + " " + str(total_acc) + " " + str(detect_rate) +
                     " " + str(sum(distance_0)/len(distance_0))+" "+str(sum(distance_1)/len(distance_1))+"\n")
    save_data("./result/"+args.save_name+'/data', distances, test_labels, epoch)

    return auc, train_feature_space

def main(args):
    try:
        os.mkdir("./result")
    except:
        pass
    try:
        os.mkdir(r"./result/" + args.save_name)
        os.mkdir(r"./result/"+args.save_name+'/data')
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
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    #train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    train_loader, test_loader = utils.get_loaders_new(dataset=args.dataset, batch_size = args.batch_size,
                                                  )
    print(type(train_loader.dataset.data))
    print(type(test_loader.dataset.data))

    #print(train_loader.dataset.targets)
    #sys.exit(1)
    train_model(model, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    #parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_name', default='abc', help='where to store result')

    args = parser.parse_args()

    main(args)
