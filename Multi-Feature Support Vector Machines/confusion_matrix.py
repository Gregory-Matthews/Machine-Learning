"""
Gregory Matthews
CS 383: Assignment 5 - Confusion Matrix
Last Modified: 2/26/2018
"""
import numpy as np
import multiclass_svm as svm
np.set_printoptions(precision=4) # Set numpy array precision
K = 3

# Initialize confusion matrix
confusion_matrix = np.zeros((K,K))

# Compute Confusion Matrix
for i, pred in enumerate(svm.predictions):
    confusion_matrix[svm.test_label[i]-1, np.argmax(pred)]+=1

confusion_matrix /= svm.predictions[:,0].size

# Printing Confusion Matrix
print confusion_matrix


