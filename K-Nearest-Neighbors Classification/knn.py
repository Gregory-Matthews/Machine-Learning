"""
Gregory Matthews
CS 383: Assignment 4 - k-Nearest Neighbors (KNN)
Last Modified: 2/20/2018
"""
import csv
import numpy as np
from scipy import stats as sp
np.random.seed(0)
np.set_printoptions(suppress=True)
k = 5  # Initializing value of k

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(X.shape[0]-1))

# Reading in data
with open('spambase.data', 'rb') as file:
    input = csv.reader(file)
    data = np.asmatrix(list(input)).astype(float)

# Randomizing Dataset
np.random.shuffle(data)

# Splitting features and label
X = data[:,:-1]
Y = data[:,-1]

N = X.shape[0] # Observations / Feature
D = X.shape[1] # Num of Features

# Splitting into training (2/3) and testing sets(1/3)
index = int(np.ceil(2*N/3))
train = X[:index]
train_label = Y[:index]
test = X[index:]
test_label = Y[index:]

# Standardizing data
for i in range(D):
    mean = np.mean(train[:,i])
    std = standard_dev(train[:,i], mean)
    train[:,i] = (train[:,i] - mean) / std
    test[:, i] = (test[:, i] - mean) / std

classification = np.zeros((test.shape[0],1))

# Perform k-Nearest Neighbor Classification
for i, test_obs in enumerate(test):
    # Holds distance of all neighbors to current test observation (L1 Manhattan Distance)
    neighbor_dist = np.sum(np.abs(train-test_obs), axis=1)

    # Find indices of k nearest neighbors by partitioning and selecting k-smallest values
    idx = np.argpartition(neighbor_dist.transpose(), k)[:,:k]

    # Get labels corresponding to the k nearest neighbors and find mode
    label = sp.mode(train_label[idx], axis=1)

    # Add mode of k-NN labels to classification corresponding to ith test observation
    classification[i] = int(label[0])

# Computing precision, recall, f1, and accuracy
tp, tn, fp, fn = 0.,0.,0.,0.
for i,_ in enumerate(test_label):
    if classification[i] == test_label[i]:
        if classification[i] == 1: tp+=1
        if classification[i] == 0: tn+=1
    else:
        if classification[i] == 1: fp+=1
        if classification[i] == 0: fn+=1

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+tn+fp+fn)

print "precision: %.4f" % precision
print "recall: %.4f" % recall
print "f1 measure: %.4f" % f1
print "accuracy: %.4f" % accuracy