"""
Gregory Matthews
CS 383: Assignment 5 - Multi-Class Support Vector Machines
Last Modified: 2/26/2018
"""
from sklearn import svm
import csv
import numpy as np
from scipy import stats as sp
np.random.seed(0)
np.set_printoptions(suppress=True)
K = 3 # Number of Classes


# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2.))/(X.shape[0]-1))

# Reading in data
with open('CTG.csv', 'rb') as file:
    input = csv.reader(file)
    data = np.asmatrix(list(input)[2:]).astype(float)

# Randomizing Dataset
np.random.shuffle(data)

# Splitting features and label
X = data[:,:-2]
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
    train[:,i] = (train[:, i] - mean) / std
    test[:, i] = (test[:, i] - mean) / std

# Stores class predictions for each test observation & # of predictions to 1 of 3 classes
predictions = np.zeros((test_label.shape[0], 3))

# Training and Predicting using binary classifiers (one vs. one) SVM
for i in range(K-1):
    for j in range(i+1, K):

        # Only use observations for training that have current binary classifier labels
        train_binary = np.empty((0, D))
        train_binary_label = np.array([])
        for index, label in enumerate(train_label):
            if label == i+1 or label == j+1:
                train_binary = np.append(train_binary, train[index], axis=0)
                train_binary_label = np.append(train_binary_label, label)

        # Train binary classifier
        classification = svm.SVC()
        classification.fit(train_binary, train_binary_label)

        # Test current classifier on all test samples
        for index, sample in enumerate(test):

            # Increment class prediction into corresponding test sample index
            predictions[index][int(classification.predict(sample)[0]-1)] += 1


# Computing Accuracy
correct, incorrect = 0.,0.

# Compare prediction to actual label
for i, pred in enumerate(predictions):

    # Prediction has a tie among classes, choose randomly
    if np.all(pred) == 1:
        if np.random(pred)+1 == test_label[i]: correct +=1
        else: incorrect +=1

    # No tie, find top chosen prediction
    elif np.argmax(pred)+1 == test_label[i]: correct +=1
    else: incorrect +=1

def main():
    # Printing Accuracy
    print "accuracy: %.4f" % (correct / (correct + incorrect))

if __name__ == "__main__":
    main()



