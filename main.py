# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import time as time
import os
start = time.time()

# ------------------ For Reproducibility
np.random.seed(777)

# ------------------ Data
data = np.load('data_normalized.npy')
labels_orig = np.load('labels.npy')
m, n = data.shape
mean = data.mean(axis=0)
sigma = data.max(axis=0) - data.min(axis=0)
data = (data - mean) / sigma


# ------------------ Neural Network Parameters
n_classes = 10
a2_nodes = 70
a3_nodes = 70
max_iter = 50000
alpha = 0.02
lamda = 3
j_history = np.zeros(max_iter)

w1 = np.random.randn(a2_nodes, n+1)
w2 = np.random.randn(a3_nodes, a2_nodes+1)
w3 = np.random.randn(n_classes, a3_nodes+1)

# ------------------ One Vs All
labels = (np.arange(n_classes) == labels_orig).astype(int)

print("\n**********************************************\n")
print("Training...")
# ------------------ Training  
for i in range(max_iter):
    w1, w2, w3 = train(data, labels, w1, w2, w3, alpha=0.01, lamda=0)
    j = cost(data, labels, w1, w2, w3, lamda=lamda)
    print("\r" + "{}% |".format(int(100 * i / max_iter) + 1) + '#' * int((int(100 * i / max_iter) + 1) / 5) +
          ' ' * (20 - int((int(100 * i / max_iter) + 1) / 5)) + '|',
          end="") if not i % (max_iter / 100) else print("", end="")
    j_history[i] = j

print("\n\n**********************************************\n")
# ------------------ Analysis
train_prediction = predict(data, w1, w2, w3)
print("Accuracy on the training set= %0.2f" % (100*np.sum(train_prediction == labels_orig)/m))
print('Final Cost: ', j)

plt.plot(range(1, max_iter+1), j_history)
plt.ylabel("Cost")
plt.xlabel("No. of Iterations")
end = time.time()
print("Execution Time= %0.2fs" % (end - start))
plt.show()

#if ((100*np.sum(train_prediction == labels_orig)/m) > 75.0):
if not os.path.exists('./cost= %0.2f acc= %0.2f' % (j, 100*np.sum(train_prediction == labels_orig)/m)):
    os.makedirs('./cost= %0.2f acc= %0.2f' % (j, 100*np.sum(train_prediction == labels_orig)/m))
    os.chdir('./cost= %0.2f acc= %0.2f' % (j, 100*np.sum(train_prediction == labels_orig)/m))
    
np.save('w1', w1)
np.save('w2', w2)
np.save('w3',w3)
print("\n**********************************************\n")
    
