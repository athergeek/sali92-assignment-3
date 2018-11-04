print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn import datasets
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


def my_scorer(classifier, y_true, y_predicted):
    y_true1 = np.transpose(y_true)
    y_true1 = y_true[:, -1]

    error = math.sqrt(np.mean((1 - y_predicted)**2))
    return error


def plotKernalPCA(X, Y, n_components, dataset_name):

    pca = KernelPCA()
    pipe = Pipeline(steps=[('KernelPCA', pca)])
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'KernelPCA__n_components': n_components,
        'KernelPCA__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'],
        'KernelPCA__eigen_solver': ['auto', 'dense']
    }
    search = GridSearchCV(pipe, param_grid, iid=False, scoring=my_scorer)
    search.fit(X, Y)
    print(search.best_params_)

    # Plot the PCA spectrum
    pca.fit(X)

    fig, ax0 = plt.subplots()
    ax0.plot(pca.lambdas_, linewidth=2)
    ax0.set_ylabel('Kernal PCA - (Eigen Values)')
    ax0.set_xlabel('n_components')
    plt.title('PCA - selection for dataset%s' % dataset_name)

    ax0.axvline(search.best_estimator_.named_steps['KernelPCA'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))
    plt.show()
