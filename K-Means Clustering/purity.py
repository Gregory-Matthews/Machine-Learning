"""
Gregory Matthews
CS 383: Assignment 2 - Clustering
Last Modified: 2/2/2018
"""
import numpy as np
from clustering import *

# Solving Purity
count = 0
N_ij = np.zeros((2,2)) # Number of instances of label j in Cluster i
for index, x in enumerate(x_data67):
    for val in clusters[0]:
        if (x == val).all():
            if y_data[index] == -1:
                N_ij[0,0] += 1
            elif y_data[index] == 1:
                N_ij[0,1] += 1
            break

    for val in clusters[1]:
        if (x == val).all():
            if y_data[index] == -1:
                N_ij[1,0] += 1
            elif y_data[index] == 1:
                N_ij[1,1] += 1
            break

total = 0
for i, Ci in enumerate(N_ij):
    purity = np.amax(N_ij[i])/len(clusters[i])
    print "Purity of Cluster " + str(i) + ": " + str(purity)

    total += len(clusters[i]) * purity

total /= N_ij.sum()
print "Total Purity: " + str(total)
