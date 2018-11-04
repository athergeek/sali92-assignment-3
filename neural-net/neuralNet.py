import numpy as np
import pandas as pd
import sys
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from time import time

"""
    Neural Net classifier.

     Parameters
     ----------
     datasetPath : Path to the dataset.
     """


def NeuralNet(X, Y, scale=False):
    # Check size of the dataset
    # print(dataset.size)

    # Chceck shapre of the dataset
    # print('dataset.shape::: ', dataset.shape)

    # SPlit dataset into 3-folds 70% traning, 20% validation, 10% test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=.10, random_state=17)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=.20, random_state=17)

    # print('X_train.shape ::: ', X_train.shape)
    # Chceck shapre of the trainig and validation folds
    # print('X_validation.shape ::: ', X_validation.shape)
    # print('X_test.shape ::: ', X_test.shape)

    # Train Neural Net calssifier on training data.
    ##  {'learning_rate': 'invscaling', 'solver': 'lbfgs', 'activation': 'logistic', 'hidden_layer_sizes': 8, 'alpha': 0.1}

    classifier = MLPClassifier(activation='logistic', alpha=0.1, batch_size='auto',
                               beta_1=0.9, beta_2=0.999, early_stopping=False,
                               epsilon=1e-08, hidden_layer_sizes=8, learning_rate='invscaling',
                               learning_rate_init=0.001, max_iter=200, momentum=0.9,
                               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                               warm_start=True)

    # classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #                            beta_1=0.9, beta_2=0.999, early_stopping=False,
    #                            epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
    #                            learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #                            nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #                            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #                            warm_start=False)

    if scale:
        print("Executing Standard Feature Scaling .....")
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # apply same transformation to test data
        X_test = scaler.transform(X_test)

    previousTime = time()
    classifier.fit(X_train, Y_train)
    afterTime = time()
    print("Training Time:", round((afterTime - previousTime), 3), "s")

    accuracy = classifier.score(X_train, Y_train)
    print("Training Accuracy: ", accuracy)

    # Validate NN classifier on validation data. Check the accuracy
    validationTimeStart = time()
    classifier.predict(X_validation)
    validationTimeEnd = time()
    print("Validation Time:", round(
        (validationTimeEnd - validationTimeStart), 3), "s")

    accuracy = classifier.score(X_validation, Y_validation)
    print("Validation Accuracy: ", accuracy)

    # Use NN classifier on test data. Check the accuracy
    predTimeStart = time()
    classifier.predict(X_test)
    predTimeEnd = time()
    print("Prediction Time:", round((predTimeEnd - predTimeStart), 3), "s")

    accuracy = classifier.score(X_test, Y_test)
    print("Prediction Accuracy: ", accuracy)
