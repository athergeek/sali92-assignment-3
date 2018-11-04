
import numpy as np
from shared.silhouetteAnalysis import silhouetteAnalysis
from shared.benchKMeans import benchKMeans
from sklearn.cluster import KMeans

datasetPath1 = './datasets/abalone.csv'
dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X = dataset1[:, 0: 7]
y = dataset1[:, -1]

# range_n_clusters = [2, 4, 6, 8, 10, ]
range_n_clusters = [2, 25]
range_n_iterations = [10, 100, 500, 1000]
# # range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30]

for n_clusters in range_n_clusters:
    for n_iterations in range_n_iterations:
        print("clusters: %d, iterations: %d" % (n_clusters, n_iterations))
        print(80 * '_')
        print('init\t\ttime\tinertia\tcalinski\tsilhouette')
        benchKMeans(KMeans(init='k-means++', n_clusters=n_clusters,
                           n_init=n_iterations), name="k-means++", data=X, actual_labels=y)
        benchKMeans(KMeans(init='random', n_clusters=n_clusters, n_init=n_iterations),
                    name="random", data=X, actual_labels=y)
        print(80 * '_')


# silhouetteAnalysis with charts
# silhouetteAnalysis(X, y, 2, dataset_name='abalone')
# silhouetteAnalysis(X, y, 10, dataset_name='abalone')
