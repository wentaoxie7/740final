
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
#data = np.loadtxt('./result/cifar100_L2/data/49.txt')
data = np.loadtxt('./result/cifar100_oe/data/0.txt')

# Split the data into two arrays based on the labels
label_0 = data[data[:, 1] == 0][:, 0]
label_1 = data[data[:, 1] == 1][:, 0]
print(label_0.mean())
print(label_1.mean())
# Plot the density distributions for the two labels
plt.hist(label_0, bins = 1000, density=True, alpha=0.5, label='Normal images',range = (0,5))
plt.hist(label_1, bins = 1000, density=True, alpha=0.5, label='Anomaly images',range = (0,5))

# Add labels and title
plt.xlabel('Anomaly Score')
#lt.xlim((0,25))
plt.ylabel('Density')
plt.title('Density Distribution of Anomaly Scores on cifar 100; CPP')

# Show the plot
plt.legend()
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a normal distribution
x = np.random.normal(0, 1, 200)

# Plot a histogram of the data
plt.hist(x, bins=80, density=True)
plt.xlabel('Score')
plt.ylabel('Probability')
plt.title('Histogram of Scores')
plt.show()'''