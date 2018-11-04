import sys
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from shared.plotLearningCurve import plotLearningCurve
from shared.plotValidationCurve import plotValidationCurve

datasetPath1 = './datasets/abalone.csv'
dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X1 = dataset1[:, 0: 7]
Y1 = dataset1[:, -1]

# Best Params for dataset-1 (Bio-Degradation) ::::  {'learning_rate': 'invscaling', 'solver': 'lbfgs', 'activation': 'logistic', 'hidden_layer_sizes': 8, 'alpha': 0.1}
title = "Learning Curves (Neural Net, activation=logistic, alpha=0.1, batch_size=auto "
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = MLPClassifier(activation='logistic', alpha=0.1, batch_size='auto',
                          beta_1=0.9, beta_2=0.999, early_stopping=False,
                          epsilon=1e-08, hidden_layer_sizes=8, learning_rate='invscaling',
                          learning_rate_init=0.001, max_iter=200, momentum=0.9,
                          nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                          solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                          warm_start=True)
plt = plotLearningCurve(estimator, title, X1, Y1, (0.1, 1.01), cv=cv, n_jobs=1)
plt.show()


plt = plotValidationCurve(MLPClassifier(), "Validation Curve For Neural Net",
                          X1, Y1, "alpha", [0.0001, 0.001, 0.01, 0.1], "alpha param value")
plt.show()
