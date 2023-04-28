import torch
import numpy as np


distance_0 = []
distance_1 = []
distances = []
test_labels = []
with open(r"./result/cifar100_oe/data/0.txt","r") as f:
    ls = f.readlines()
    for l in ls:
        data = float(l.split(" ")[0])
        label = int(l.split(" ")[1])
        distances.append(data)
        if label == 1:
            distance_1.append(data)
            test_labels.append(1)
        else:
            distance_0.append(data)
            test_labels.append(0)



distance_0t = torch.tensor(distance_0)
distance_1t = torch.tensor(distance_1)

mean_distance = torch.mean(distance_0t)
mean_1 = torch.mean(distance_1t)
stddev_distance = torch.std(distance_0t)
#threshold = mean_distance - 0.008* stddev_distance
distance_0.sort()
#print(distance_0[9000:9010])
threshold = distance_0[9500]
pred_labels = np.where(torch.tensor(distances) > threshold, 1, 0)
total_acc = (np.sum((pred_labels==0) & ((np.array(test_labels))==0))+ np.sum(pred_labels[(np.array(test_labels))==1]))/len(test_labels)
detect_rate = np.sum(pred_labels[(np.array(test_labels))==1])
print(mean_distance, mean_1, threshold, total_acc, detect_rate)
print(len(distance_0),len(distance_1),len(test_labels))
