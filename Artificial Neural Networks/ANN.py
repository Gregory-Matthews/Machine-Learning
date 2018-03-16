"""
Gregory Matthews
CS 383: Assignment 6 - Artificial Neural Network
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

Accuracy = [] # Container for training accuracy
M = 20 # Number of hidden layer nodes
K = 1 # Number of output layer nodes
learning_rate = 0.5 # Learning Parameter
epochs = 1000 # Training Iterations
threshold = 0.5 # Threshold for predictions

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(X.shape[0]-1))

# Sigmoid Activation Function
def g(z):
    return 1/(1+np.exp(-z))

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

# Add bias feature to training and testing data
train = np.append(np.ones((train.shape[0], 1)), train, axis=1)
test = np.append(np.ones((test.shape[0], 1)), test, axis=1)


# Initializing Beta and Theta over random range [-1,1]
Beta = np.random.uniform(-1., 1., size=(D+1)*M)
Beta = Beta.reshape((D+1,M))
Theta = np.random.uniform(-1., 1., size=M*K)
Theta = Theta.reshape((M,K))

"""Training Neural Network"""

for _ in range(epochs):
    # Compute output of hidden layer node
    H = g(train*Beta)

    # Compute output of output layer node
    O = g(H*Theta)

    # Compute Accuracy of training epoch
    correct = 0
    for i, o_i in enumerate(O):
        # Compare against threshold
        if o_i > threshold:
            pred = 1
        else: pred = 0
        if pred == train_label[i]: correct +=1.

    # Add training accuracy
    Accuracy.append(correct/O.shape[0])

    # Update Weights
    delta_out = train_label - O
    Theta += (learning_rate/N) * H.transpose() * delta_out

    delta_hidden = np.multiply(np.multiply(delta_out*Theta.transpose(), H), (1-H))
    Beta += (learning_rate/N) * train.transpose() * delta_hidden

mplot.plot(Accuracy, c='b')
mplot.title('ANN Training Accuracy')
mplot.ylabel('% Accuracy')
mplot.xlabel('# of Training Iterations')
mplot.savefig('ANN_Training_Accuracy.png')


"""Testing Neural Network"""
def main():
    # Compute output of hidden layer node
    H = g(test * Beta)

    # Compute output of output layer node
    O = g(H * Theta)

    # Compute Accuracy of test data
    correct = 0
    for i, o_i in enumerate(O):
        # Compare against threshold
        if o_i > threshold:
            pred = 1
        else:
            pred = 0
        if pred == test_label[i]: correct += 1.

    # Print testing accuracy
    print "Testing Accuracy = " +  str(correct/O.shape[0]) + "%"


if __name__ == "__main__":
    main()

