
import numpy as np
from shared.silhouetteAnalysis import silhouetteAnalysis
from shared.benchKMeans import benchKMeans
from shared.plotKernalPCA import plotKernalPCA
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

datasetPath1 = './datasets/qsar-biodeg.csv'
dataset1 = np.loadtxt(datasetPath1, delimiter=',', skiprows=1)

X = dataset1[:, 0: 41]
y = dataset1[:, -1]

components_to_keep = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]

plotKernalPCA(X, y, components_to_keep, dataset_name=' Qsar-Biodeg')


datasetPath2 = './datasets/abalone.csv'
dataset2 = np.loadtxt(datasetPath2, delimiter=',', skiprows=1)

X2 = dataset2[:, 0: 7]
y2 = dataset2[:, -1]
components_to_keep = [2, 4, 6, 7]
plotKernalPCA(X2, y2, components_to_keep, dataset_name=' Abalone')
