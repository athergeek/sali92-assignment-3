print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def plotPCA(X, Y, n_components, dataset_name):

    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca)])
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': n_components
    }
    search = GridSearchCV(pipe, param_grid, iid=False)
    search.fit(X, Y)
    print(search.best_params_)

    # Plot the PCA spectrum
    pca.fit(X)

    fig, ax0 = plt.subplots()
    ax0.plot(pca.explained_variance_ratio_, linewidth=2)
    ax0.set_ylabel('PCA explained variance (Eigen Values)')
    ax0.set_xlabel('n_components')
    plt.title('PCA - selection for dataset%s' % dataset_name)

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))
    plt.show()
