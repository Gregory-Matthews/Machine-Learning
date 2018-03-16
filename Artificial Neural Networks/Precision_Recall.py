"""
Gregory Matthews
CS 383: Assignment 6 - Artificial Neural Network, Precision and Recall
Last Modified: 3/2/2018
"""
import csv
import numpy as np
from scipy import stats as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mplot
np.random.seed(0)
np.set_printoptions(suppress=True)

import ANN as NN

precision = []
recall = []

# Compute output of hidden layer node
NN.H = NN.g(NN.test * NN.Beta)

# Compute output of output layer node
NN.O = NN.g(NN.H * NN.Theta)

# Calculating Precision and Recall
for threshold in range(0, 11):
    # Compute Precision and Recall
    tp, tn, fp, fn = 0., 0., 0., 0.
    for i, o_i in enumerate(NN.O):
        # Compare against threshold
        if o_i > threshold/10.:
            pred = 1
        else:
            pred = 0

        if pred == NN.test_label[i]:
            if pred == 1: tp +=1
            elif pred == 0: tn += 1
        else:
            if pred == 1: fp +=1
            elif pred == 0: fn += 1
    # Check for division by 0 case
    if (tp+fp) == 0:
        precision.append(1.0)
    else:
        precision.append(tp / (tp + fp))

    recall.append(tp / (tp + fn))

mplot.figure(2)
mplot.plot(precision, recall, marker='o', c='b')
mplot.title('Spam Detection')
mplot.xlabel('Precision')
mplot.ylabel('Recall')
mplot.savefig('Precision_Recall.png')
