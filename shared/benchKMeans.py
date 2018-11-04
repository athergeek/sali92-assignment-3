from time import time
from sklearn import metrics


def benchKMeans(estimator, name, data, actual_labels, metric='euclidean', sample_size=300):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t\t%.3f'
          # print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             #  metrics.homogeneity_score(actual_labels, estimator.labels_),
             #  metrics.completeness_score(actual_labels, estimator.labels_),
             #  metrics.v_measure_score(actual_labels, estimator.labels_),
             #  metrics.adjusted_rand_score(actual_labels, estimator.labels_),
             #  metrics.adjusted_mutual_info_score(
             #   actual_labels,  estimator.labels_),
             metrics.calinski_harabaz_score(data, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric=metric,
                                      sample_size=sample_size)))
