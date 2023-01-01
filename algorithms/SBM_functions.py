import math
import sys
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import warnings

from common.distance import euclidean_point_distance
from common.neighbourhood import get_valid_neighbours

sys.setrecursionlimit(1000000)
parallel_backend('threading')
num_cores = multiprocessing.cpu_count() - 2

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO
def valid_center(value):
    return True  # TODO set the min value acceptable


def check_maxima(array, point):
    """
    Check whether the chunk with the coordinates "point" is bigger than all its neighbours
    :param array: matrix - an array of the values in each chunk
    :param point: vector - the coordinates of the chunk we are looking at

    :returns : boolean - whether or not it is a maxima
    """
    neighbours = get_valid_neighbours(point, np.shape(array))


    for neighbour in neighbours:
        if array[tuple(neighbour)] > array[point]:
            return False
    # fastest way but you need to convert array of arrays to array of tuples
    # neighbours = np.apply_along_axis(tuple, 0, neighbours)
    # this doesnt work because python retarded
    # print(np.all(array[neighbours] < array[point]))

    return True


def find_cluster_centers(array, threshold=5):
    """
    Search through the matrix of chunks to find the cluster centers
    :param array: matrix - an array of the values in each chunk
    :param threshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

    :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
    """
    clusterCenters = []

    for index, value in np.ndenumerate(array):
        if value >= threshold and check_maxima(array, index):  # TODO exclude neighbour centers
            clusterCenters.append(index)

    return clusterCenters


def get_dropoff(ndArray, location):
    neighbours = get_valid_neighbours(location, np.shape(ndArray))
    dropoff = 0
    for neighbour in neighbours:
        neighbourLocation = tuple(neighbour)
        dropoff += ((ndArray[location] - ndArray[neighbourLocation]) ** 2) / ndArray[location]
    if dropoff > 0:
        return math.sqrt(dropoff / len(neighbours))
    return 0


def get_strength(ndArray, clusterCenter, questionPoint, expansionPoint):
    dist = euclidean_point_distance(clusterCenter, questionPoint)
    # strength = ndArray[expansionPoint] / ndArray[questionPoint] / dist

    # TODO
    strength = ndArray[questionPoint] / dist / ndArray[clusterCenter]

    return strength


def expand_cluster_center(array, start, labels, currentLabel, clusterCenters, version=1):  # TODO
    """
    Expansion
    :param array: matrix - an array of the values in each chunk
    :param start: tuple - the coordinates of the chunk where the expansion starts (current cluster center)
    :param labels: matrix - the labels array
    :param currentLabel: integer - the label of the current cluster center
    :param clusterCenters: vector - vector of all the cluster centers, each containing n-dimensions
    :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)

    :returns labels: matrix - updated matrix of labels after expansion and conflict solve
    """
    visited = np.zeros_like(array, dtype=bool)
    expansionQueue = []
    if labels[start] == 0:
        expansionQueue.append(start)
        labels[start] = currentLabel
    # else:
    #     oldLabel = labels[start]
    #     disRez = disambiguate(array,
    #                          start,
    #                          clusterCenters[currentLabel - 1],
    #                          clusterCenters[oldLabel - 1])
    #     if disRez == 1:
    #         labels[start] = currentLabel
    #         expansionQueue.append(start)
    #     elif disRez == 11:
    #         labels[labels == oldLabel] = currentLabel
    #         expansionQueue.append(start)
    #     elif disRez == 22:
    #         labels[labels == currentLabel] = oldLabel
    #         currentLabel = oldLabel
    #         expansionQueue.append(start)

    visited[start] = True

    dropoff = get_dropoff(array, start)

    while expansionQueue:
        point = expansionQueue.pop(0)
        neighbours = get_valid_neighbours(point, np.shape(array))

        for neighbour in neighbours:
            location = tuple(neighbour)
            if version == 1:
                number = dropoff * math.sqrt(euclidean_point_distance(start, location))
            elif version == 2:
                number = math.floor(math.sqrt(dropoff * euclidean_point_distance(start, location)))

            if array[location] == 0:
                pass
            if (not visited[location]) and (number < array[location] <= array[point]):
                visited[location] = True
                if labels[location] == currentLabel:
                    expansionQueue.append(location)
                elif labels[location] == 0:
                    expansionQueue.append(location)
                    labels[location] = currentLabel
                else:
                    if version == 0:
                        labels[location] = -1
                    else:
                        oldLabel = labels[location]
                        disRez = disambiguate(array,
                                              location,
                                              point,
                                              clusterCenters[currentLabel - 1],
                                              clusterCenters[oldLabel - 1],
                                              version)
                        # print(currentLabel, oldLabel, disRez)
                        if disRez == 1:
                            labels[location] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 2 and version == 2:
                            labels[location] = oldLabel
                            expansionQueue.append(location)
                        elif disRez == 11:
                            # current label wins
                            labels[labels == oldLabel] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 22:
                            # old label wins
                            labels[labels == currentLabel] = oldLabel
                            currentLabel = oldLabel
                            expansionQueue.append(location)

    return labels


def disambiguate(array, questionPoint, expansionPoint, clusterCenter1, clusterCenter2, version):
    """
    Disambiguation of the cluster of a chunk based on the parameters
    :param array: matrix - an array of the values in each chunk
    :param questionPoint: tuple - the coordinates of the chunk toward which the expansion is going
    :param expansionPoint: tuple - the coordinates of the chunk from which the expansion is going
    :param clusterCenter1: tuple - the coordinates of the chunk of the first cluster center
    :param clusterCenter2: tuple - the coordinates of the chunk of the second cluster center
    :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)

    :returns : integer - representing the approach to disambiguation
    """
    # CHOOSE CLUSTER FOR ALREADY ASSIGNED POINT
    # usually wont get to this
    if (clusterCenter1 == questionPoint) or (clusterCenter2 == questionPoint):
        # here the point in question as already been assigned to one cluster, but another is trying to accumulate it
        # we check which of the 2 has a count and merged them
        if array[clusterCenter1] > array[clusterCenter2]:
            return 11
        else:
            return 22

    # MERGE
    # cluster 2 was expanded first, but it is actually connected to a bigger cluster
    if array[clusterCenter2] == array[questionPoint]:
        return 11
    if version == 2:
        # cluster 1 was expanded first, but it is actually connected to a bigger cluster
        if array[clusterCenter1] == array[questionPoint]:
            return 22

    # XANNY
    if version == 1:
        distanceToC1 = euclidean_point_distance(questionPoint, clusterCenter1)
        distanceToC2 = euclidean_point_distance(questionPoint, clusterCenter2)
        pointStrength = array[questionPoint]

        c1Strength = array[clusterCenter1] / pointStrength - get_dropoff(array, clusterCenter1) * distanceToC1
        c2Strength = array[clusterCenter2] / pointStrength - get_dropoff(array, clusterCenter2) * distanceToC2

    # RICI
    elif version == 2:
        c1Strength = get_strength(array, clusterCenter1, questionPoint, expansionPoint)
        c2Strength = get_strength(array, clusterCenter2, questionPoint, expansionPoint)

    # RICI VERSION
    # neighbours = getNeighbours(questionPoint, np.shape(array))
    # maxN = 0
    # for n in neighbours:
    #     if labels[tuple(n)] == oldLabel and array[tuple(n)] > maxN:
    #         maxN = array[tuple(n)]
    #
    # c1Strength = array[questionPoint] / array[cluster1]
    # c2Strength = array[questionPoint] / array[cluster2]

    # distanceToC1 = distance(questionPoint, clusterCenter1)
    # distanceToC2 = distance(questionPoint, clusterCenter2)
    # pointStrength = array[questionPoint]
    # c1Strength = array[cluster1]*distanceToC1/(array[cluster1] - pointStrength)
    # c2Strength = array[cluster2]*distanceToC2/(array[cluster2] - pointStrength)
    # c2Strength = (array[cluster2] / pointStrength) / distanceToC2

    # if (abs(c1Strength - c2Strength) < threshold):
    #     return 0
    if c1Strength > c2Strength:
        return 1
    else:
        return 2


def chunkify_sequential(X, pn):
    """
    Transforms the points into a matrix of integers by gridding the dataset and counting the points in each item of the grid
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows

    :returns nArray: matrix - each entry is a an integer that contains the number of points in a square-like partition
    """
    nrDim = np.shape(X)[1]
    nArray = np.zeros((pn,) * nrDim, dtype=int)

    for point in X:
        if np.all(point < pn):
            location = tuple(np.floor(point).astype(int))
            nArray[location] += 1
        else:  # TODO
            # print(point)
            pass
    return nArray


def chunkify_numpy(X, pn):
    nrDim = np.shape(X)[1]
    nArray = np.zeros((pn,) * nrDim, dtype=int)

    # floor is done before the iteration over a for because it will be faster using numpy
    R = np.floor(X).astype(int)

    # represents the if from before, instead applied on the whole nd-array
    # (this way removing outliers, points that contain pn <- unavoidable by normalization)
    R = R[np.all(R < pn, axis=1)]

    # TODO FutureWarning R will have to be tuple

    # adding (the counts) in the nArray using the coordinates of R
    R = np.transpose(R).tolist()
    np.add.at(nArray, R, 1)

    return nArray


def chunkify_parallel(X, pn, nrThreads=num_cores):
    """
    Multi-threaded version of the chunkify function
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param nrThreads: integer - optional, the number of threads the chunkification runs on

    :returns finalArray: matrix - each entry is a an integer that contains the number of points in a square-like partition
    """
    splittedX = np.array_split(X, nrThreads)

    results = Parallel(n_jobs=nrThreads)(delayed(chunkify_sequential)(x, pn) for x in splittedX)

    finalArray = sum(results)

    return finalArray


def dechunkify_parallel(X, labelsArray, pn, nrThreads=num_cores):
    """
    Multi-threaded version of the dechunkify function
    :param X: matrix - the points of the dataset
    :param labelsArray: matrix - contains the labels of each of the partitions/chunks
    :param pn: integer - the number of partitions on columns and rows

    :returns finalLabels: vector - the labels of the points
    """
    splittedX = np.array_split(X, nrThreads)

    results = Parallel(n_jobs=nrThreads)(delayed(dechunkify_sequential)(x, labelsArray, pn) for x in splittedX)

    finalLabels = np.concatenate(results, axis=0)
    return finalLabels


def dechunkify_numpy(X, labelsArray, pn):
    # floor is done before the iteration over a for because it will be faster using numpy
    R = np.floor(X).astype(int)

    # (this way removing outliers, points that contain pn <- unavoidable by normalization)
    R[R >= pn] = pn - 1

    # TODO FutureWarning R will have to be tuple

    # get the coordinates in the nArray (of all points) in Q and use that to set the labels of all the points in the dataset
    Q = np.transpose(R).tolist()
    pointLabels = labelsArray[Q]

    return pointLabels


def dechunkify_sequential(X, labelsArray, pn):
    """
    Transforms the labels of the chunks into the labels of each of the points
    :param X: matrix - the points of the dataset
    :param labelsArray: matrix - contains the labels of each of the partitions/chunks
    :param pn: integer - the number of partitions on columns and rows

    :returns finalLabels: vector - the labels of the points
    """
    pointLabels = np.zeros(len(X), dtype=int)

    # import threading
    # print("Thread ID[{}] gets {}".format(threading.get_ident(), X[:1]))

    for index in range(0, len(X)):
        point = X[index]
        if np.all(point < pn):
            location = tuple(np.floor(point).astype(int))
            pointLabels[index] = labelsArray[location]
        else:  # TODO
            pointLabels[index] = -1

    return pointLabels


