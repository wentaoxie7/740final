import numpy as np
import torch

distance_0 = [0.1,0.2,0.3,0.9,0.8]
distance_1 = [0.4,0.5,0.6,0.7,1.0]
distances = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
test_labels = np.array([0,0,0,1,1,1,1,0,0,1])
distance_0t = torch.tensor(distance_0)
distance_1t = torch.tensor(distance_1)
mean_distance = torch.mean(distance_0t)
stddev_distance = torch.std(distance_0t)
threshold = mean_distance + 0 * stddev_distance

pred_labels = np.where(torch.tensor(distances) > threshold, 1, 0)
print(pred_labels)
print(test_labels)
total_acc = np.sum((pred_labels==0) & (test_labels==0))
detect_rate = np.sum(pred_labels[test_labels==1])
print(mean_distance)
print(stddev_distance)
print(threshold)
print(total_acc)
print(detect_rate)