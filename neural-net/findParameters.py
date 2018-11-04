import numpy as np
from time import time
from neuralNetGridSearch import neuralNetGridSearch

from shared.checkDataset import datasetDetails


datasetPath1 = './datasets/abalone.csv'

dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X1 = dataset1[:, 0: 7]
Y1 = dataset1[:, -1]

datasetDetails(dataset1, X1, Y1)


previousTime = time()
# Best Params for dataset-1 (Bio-Degradation) ::::  {'learning_rate': 'invscaling', 'solver': 'lbfgs', 'activation': 'logistic', 'hidden_layer_sizes': 8, 'alpha': 0.1}
# Parameter finding Time: 2175.177 s
params = neuralNetGridSearch(X1, Y1, 2)
print('Best Params for dataset-1 (Bio-Degradation) :::: ', params)
afterTime = time()
print("Parameter finding Time:", round((afterTime - previousTime), 3), "s")
