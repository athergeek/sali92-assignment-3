
import matplotlib.pyplot as plt


def plotDataset(label, dataset):
    plt.figure(1)
    plt.clf()
    plt.plot(dataset[:, 0], dataset[:, 1], 'k.', markersize=2)
    # plt.plot(dataset[:, 0], dataset[:, 1], dataset[:, 2],
    #          dataset[:, 3], dataset[:, 4], dataset[:, 5], dataset[:, 6], dataset[:, 7], 'k.', markersize=2)
    plt.title(label)
    plt.show()
