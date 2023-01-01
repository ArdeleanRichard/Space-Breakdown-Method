import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# constants for particular datasets
from constants import dataFiles, dataName, DATA_FOLDER_PATH

kmeansValues = [15, 15, 8, 6, 20]
epsValues = [27000, 45000, 18000, 0.5, 0.1]
pn = 25


def load_synthetic_data(datasetNumber):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)

    :returns None

    # datasetNumber = 1 => S1
    # datasetNumber = 2 => S2
    # datasetNumber = 3 => U
    # datasetNumber = 4 => UO - neural simulated data from gen_simulated_data
    """
    print("DATASET: " + dataName[datasetNumber])

    if datasetNumber < 3:
        X = np.genfromtxt(DATA_FOLDER_PATH + '/CLUSTERING/' + dataFiles[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif datasetNumber == 3:
        X, y = getGenData()
    else:
        X, y = load_real_data()

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1



def load_real_data(chance=False):
    # Importing the dataset
    data = pd.read_csv(DATA_FOLDER_PATH + '/CLUSTERING/real_data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.hstack((c1, c2, c3))

    if chance == True:
        chanceKeep = 1
        keep = np.random.choice(2, len(X), p=[1 - chanceKeep, chanceKeep])
        keep = keep == 1
        X = X[keep]

    return X, None


def generate_simulated_data(avgPoints=250):
    np.random.seed(0)
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)

    C3 = [1, -2] + .2 * np.random.randn(avgPoints * 5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)

    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)

    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)

    X = np.vstack((C5, C1, C2, C3, C4, C6))

    c1Labels = np.full(len(C1), 1)
    c2Labels = np.full(len(C2), 2)
    c3Labels = np.full(len(C3), 3)
    c4Labels = np.full(len(C4), 4)
    c5Labels = np.full(len(C5), 5)
    c6Labels = np.full(len(C6), 6)

    y = np.hstack((c5Labels, c1Labels, c2Labels, c3Labels, c4Labels, c6Labels))
    return X, y


def getGenData(plotFig=False):
    np.random.seed(0)
    avgPoints = 250
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)
    # L4 = np.full(len(C4), 1).reshape((len(C4), 1))

    C3 = [1, -2] + .2 * np.random.randn(avgPoints * 5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)
    # L5 = np.full(len(C5), 2)

    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)
    # L2 = np.full(len(C2), 1).reshape((len(C2), 1))

    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)
    # L6 = np.full(len(C6), 3).reshape((len(C6), 1))

    if plotFig:
        plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
        plt.plot(C2[:, 0], C2[:, 1], 'r.', alpha=0.3)
        plt.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
        plt.plot(C4[:, 0], C4[:, 1], 'c.', alpha=0.3)
        plt.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
        plt.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)
        # plt.figure()
        # plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
        # plt.plot(C2[:, 0], C2[:, 1], 'b.', alpha=0.3)
        # plt.plot(C3[:, 0], C3[:, 1], 'b.', alpha=0.3)
        # plt.plot(C4[:, 0], C4[:, 1], 'b.', alpha=0.3)
        # plt.plot(C5[:, 0], C5[:, 1], 'b.', alpha=0.3)
        # plt.plot(C6[:, 0], C6[:, 1], 'b.', alpha=0.3)

        plt.show()
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    c1Labels = np.full(len(C1), 1)
    c2Labels = np.full(len(C2), 2)
    c3Labels = np.full(len(C3), 3)
    c4Labels = np.full(len(C4), 4)
    c5Labels = np.full(len(C5), 5)
    c6Labels = np.full(len(C6), 6)

    y = np.hstack((c1Labels, c2Labels, c3Labels, c4Labels, c5Labels, c6Labels))
    return X, y


def getDatasetS1():
    X = np.genfromtxt(DATA_FOLDER_PATH + "/CLUSTERING/s1_labeled.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y


def getDatasetS2():
    X = np.genfromtxt(DATA_FOLDER_PATH + "/CLUSTERING/s2_labeled.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y


def getDatasetU():
    X = np.genfromtxt(DATA_FOLDER_PATH + "/CLUSTERING/unbalance.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y



def generate_star_data(avgPoints=250):
    np.random.seed(0)
    C5 = [3, 2] + [1.0, 8] * np.random.randn(avgPoints * 4, 2)

    C1 = [3, 2] + [8, 1.0] * np.random.randn(avgPoints * 4, 2)

    X = np.vstack((C5, C1))

    return X, None

def generate_star_data2(avgPoints=250):
    np.random.seed(0)
    values = np.random.randn(avgPoints * 4, 1)
    C5 = [3, 2] + np.hstack((values, values))

    C1 = [3, 2] + np.hstack((values, -1 * values))

    X = np.vstack((C5, C1))

    return X, None