print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn import datasets
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer


def my_scorer(classifier, y_true, y_predicted):
    y_true1 = np.transpose(y_true)
    y_true1 = y_true[:, -1]

    error = np.mean((y_true1 - y_predicted))
    return error


def plotRandomProjection(X, Y, n_components, dataset_name):

    grp = GaussianRandomProjection()
    pipe = Pipeline(steps=[('grp', grp)])
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'grp__n_components': n_components
    }
    search = GridSearchCV(pipe, param_grid, iid=False,
                          scoring=my_scorer)

    # search = GridSearchCV(pipe, param_grid, iid=False,
    #                       scoring=make_scorer(f1_score))
    search.fit(X, Y)
    print(search.best_params_)

    # Plot the PCA spectrum
    grp.fit(X)

    fig, ax0 = plt.subplots()
    ax0.plot(grp.explained_variance_ratio_, linewidth=2)
    ax0.set_ylabel('Gaussian Random Projection')
    ax0.set_xlabel('n_components')
    plt.title('Gaussian Random Projection - selection for dataset%s' %
              dataset_name)

    ax0.axvline(search.best_estimator_.named_steps['grp'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))
    plt.show()
