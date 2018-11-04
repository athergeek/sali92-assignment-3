# Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier

print(__doc__)
NNEstimator = MLPClassifier(activation='logistic', alpha=0.1, batch_size='auto',
                            beta_1=0.9, beta_2=0.999, early_stopping=False,
                            epsilon=1e-08, hidden_layer_sizes=8, learning_rate='invscaling',
                            learning_rate_init=0.001, max_iter=200, momentum=0.9,
                            nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                            warm_start=True)
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', NNEstimator)
])

N_FEATURES_OPTIONS = [2, 4, 8, 15, 20, 25, 30, 35, 40]
LEARNING_RATE = ['constant', 'invscaling', 'adaptive'],
ALPHA = [0.0001, 0.001, 0.01, 0.1],
HIDDEN_LAYER_SIZES = [2, 4, 6, 8]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), FastICA(), GaussianRandomProjection(), KernelPCA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__learning_rate': ['constant', 'invscaling', 'adaptive'],
    },
]
reducer_labels = ['PCA', 'ICA', 'Random-Projection', 'KernalPCA']

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)
# digits = load_digits()

datasetPath1 = './datasets/qsar-biodeg.csv'
dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X = dataset1[:, 0: 41]
y = dataset1[:, -1]

grid.fit(X, y)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(
    len(LEARNING_RATE), -1, len(N_FEATURES_OPTIONS))
# select score for best LEARNING_RATE
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Neural Net accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()
