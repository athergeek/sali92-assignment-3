import sys
import numpy as np
from time import time
from neuralNet import NeuralNet


datasetPath1 = './datasets/abalone.csv'
dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X1 = dataset1[:, 0: 7]
Y1 = dataset1[:, -1]


print('')
print('**************************************')
print('****Dateset - Abalone ****************')
print('activation= logistic, alpha=0.1, batch_size= auto')
print('beta_1=0.9, beta_2=0.999, early_stopping=False, ')
print('epsilon=1e-08, hidden_layer_sizes=8, learning_rate=invscaling')
print('**************************************')
NeuralNet(X1, Y1)
