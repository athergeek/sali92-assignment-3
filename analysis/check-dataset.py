import numpy as np
from time import time

from shared.plotDataset import plotDataset


datasetPath = './datasets/abalone.csv'

dataset1 = np.loadtxt(datasetPath, delimiter=',')
# dataset1 = np.loadtxt(datasetPath, delimiter=',', skiprows=1)

plotDataset('Abalone', dataset1)
