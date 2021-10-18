import numpy as np
import pandas as pd
from functions.constants import dataName, dataFiles

def load_real_data():
    # Importing the dataset
    data = pd.read_csv('../data/real_data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.hstack((c1, c2, c3))
    return X, None

# datasetNumber = 1 => S1
# datasetNumber = 2 => S2
# datasetNumber = 3 => U
# datasetNumber = 4 => UO - neural simulated data from gen_simulated_data
def load_synthetic_data(datasetNumber):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)

    :returns X - data
    """

    if datasetNumber < 3:
        X = np.genfromtxt("../data/" + dataFiles[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif datasetNumber == 3:
        X, y = generate_simulated_data()

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    return X, y

def generate_simulated_data():
    np.random.seed(0)
    avgPoints = 250
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)

    C3 = [1, -2] + .2 * np.random.randn(avgPoints * 5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)

    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)

    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)

    X = np.vstack((C1, C2, C3, C4, C5, C6))

    c1Labels = np.full(len(C1), 1)
    c2Labels = np.full(len(C2), 2)
    c3Labels = np.full(len(C3), 3)
    c4Labels = np.full(len(C4), 4)
    c5Labels = np.full(len(C5), 5)
    c6Labels = np.full(len(C6), 6)

    y = np.hstack((c1Labels, c2Labels, c3Labels, c4Labels, c5Labels, c6Labels))
    return X, y