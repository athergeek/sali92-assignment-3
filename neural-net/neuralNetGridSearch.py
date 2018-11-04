from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from sklearn.model_selection import GridSearchCV

from scipy import stats
from sklearn.grid_search import RandomizedSearchCV


def neuralNetGridSearch(X, y, nfolds):

    grid_search = GridSearchCV(MLPClassifier(max_iter=500), param_grid={
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'hidden_layer_sizes': [2, 4, 6, 8],
        # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'activation': ['logistic', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam']
    })

    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def neuralNetRandomGridSearch(X, y, nn):
    rs = RandomizedSearchCV(nn, param_distributions={
        'learning_rate': stats.uniform(0.001, 0.05),
        'hidden0__units': stats.randint(4, 12),
        'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})

    rs.fit(X, y)
