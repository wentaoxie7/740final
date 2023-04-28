import matplotlib.pyplot as plt
import numpy as np
auroc1 = np.loadtxt('./result/cifar10_oe/distances.txt')
auroc2 = np.loadtxt('./result/cifar10/distances.txt')

# Split the data into two arrays based on the labels
auroc1 = auroc1[:45][:, 0]
auroc2 = auroc2[:45][:, 0]
print(auroc2)

plt.tight_layout()
plt.figure()
# plot the data points
plt.xticks(np.arange(0,46,5))
plt.xlabel('epochs')
plt.ylabel('auroc')
#plt.scatter(), , color="blue")
plt.plot(list(range(1,46)), auroc1, color='r', linewidth=2, alpha=0.6,label="CPP")
plt.plot(list(range(1,46)), auroc2, color='b', linewidth=2, alpha=0.6,label="PANDA")

plt.title('training AUROC on cifar 10')
plt.legend(loc="lower right")
plt.show()