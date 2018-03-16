"""
Gregory Matthews
CS 383: Assignment 5 - Support Vector Machines
Last Modified: 2/26/2018
"""
import csv
import numpy as np
from scipy import stats as sp
np.random.seed(0)
np.set_printoptions(suppress=True)
from sklearn import svm


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

N = X.shape[0]  # Observations / Feature
D = X.shape[1]  # Num of Features

# Splitting into training (2/3) and testing sets(1/3)
index = int(np.ceil(2*N/3.))
train = X[:index]
train_label = Y[:index].astype(int)
test = X[index:]
test_label = Y[index:].astype(int)

# Standardizing data
for i in range(D):
    mean = np.mean(train[:,i])
    std = standard_dev(train[:,i], mean)
    train[:,i] = (train[:,i] - mean) / std
    test[:, i] = (test[:, i] - mean) / std

# Train SVM model on training set
classification = svm.SVC()
classification.fit(train, train_label)

# Predict on testing set
tp, tn, fp, fn = 0.,0.,0.,0.
for i, sample in enumerate(test):
    if classification.predict(sample) == test_label[i]:
        if classification.predict(sample) == 1: tp += 1
        if classification.predict(sample) == 0: tn += 1
    else:
        if classification.predict(sample) == 1: fp += 1
        if classification.predict(sample) == 0: fn += 1

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+tn+fp+fn)

print "precision: %.4f" % precision
print "recall: %.4f" % recall
print "f1 measure: %.4f" % f1
print "accuracy: %.4f" % accuracy

