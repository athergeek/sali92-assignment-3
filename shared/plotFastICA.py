print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn import datasets
from sklearn.decomposition import FastICA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


def my_scorer(classifier, y_true, y_predicted):
    y_true1 = np.transpose(y_true)
    y_true1 = y_true[:, -1]

    error = math.sqrt(np.mean((y_true1 - y_predicted)**2))
    return error


def plotFastICA(X, Y, n_components, dataset_name):

    ica = FastICA()
    pipe = Pipeline(steps=[('FastICA', FastICA())])
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'FastICA__n_components': n_components,
    }
    search = GridSearchCV(pipe, param_grid, iid=False, scoring=my_scorer)
    search.fit(X, Y)
    print(search.best_params_)

    # Plot the ICA
    ica.fit(X)

    fig, ax0 = plt.subplots()
    ax0.plot(ica.mean_, linewidth=2)
    ax0.set_ylabel('Fast ICA - Average ')
    ax0.set_xlabel('n_components')
    plt.title('Fast ICA - selection for dataset%s' % dataset_name)

    ax0.axvline(search.best_estimator_.named_steps['FastICA'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))
    plt.show()
