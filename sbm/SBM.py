import os
import time
import math
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

from .common.distance import euclidean_point_distance
from .common.maxima import check_maxima, check_maxima_no_neighbour_maxim
from .common.neighbourhood import get_valid_neighbours
from .dataset_parsing.simulations_dataset import get_dataset_simulation_pca_2d
from .visualization import scatter_plot as sp

warnings.simplefilter(action='ignore', category=FutureWarning)



class SBM:
    def __init__(self, data, pn=25, threshold=5, version=2):
        """
        Constructor
        :param data: ndarray - the points of the dataset
        :param pn: integer - the number of partitions on columns and rows
        :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)
        :param ccThreshold: integer - the minimum number of points needed in a partition/chunk for it to be considered a possible cluster center
        """
        self.data = data
        self.chunked_data = self.data
        self.pn = pn
        self.threshold = threshold
        self.version = version

        self.nrDim = np.shape(self.data)[1]
        self.chunk_array = np.zeros((pn,) * self.nrDim, dtype=int)
        self.cluster_centers = []
        self.labels_array = np.zeros_like(self.chunk_array, dtype=int)
        self.labels = np.zeros(len(self.data), dtype=int)

    def fit(self):
        """
        Numpy parallelization version of SBM
        """
        self.preprocess_data()
        self.chunkify_numpy()
        self.find_cluster_centers()
        self.expand_all_cluster_centers()
        self.dechunkify_numpy()

    def preprocess_data(self):
        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled = X_std * (max - min) + min
        # X = fs.min_max_scaling(X, pn)
        self.data = preprocessing.MinMaxScaler((0, self.pn)).fit_transform(self.data)
        self.chunked_data = np.floor(self.data).astype(int)

    # TODO
    def valid_center(self, value):
        return True  # TODO set the min value acceptable

    def find_cluster_centers(self):
        """
        Search through the matrix of chunks to find the cluster centers
        :param array: matrix - an array of the values in each chunk
        :param threshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

        :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
        """

        for index, value in np.ndenumerate(self.chunk_array):
            if value >= self.threshold and check_maxima(self.chunk_array, index):  # TODO exclude neighbour centers
                self.cluster_centers.append(index)


    def find_cluster_centers_no_neighbours(self):
        """
        Search through the matrix of chunks to find the cluster centers
        :param array: matrix - an array of the values in each chunk
        :param threshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

        :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
        """
        for index, value in np.ndenumerate(self.chunk_array):
            if value >= self.threshold and check_maxima_no_neighbour_maxim(self.chunk_array, index, self.cluster_centers):
                self.cluster_centers.append(index)


    def get_dropoff(self, location):
        neighbours = get_valid_neighbours(location, np.shape(self.chunk_array))
        dropoff = 0
        for neighbour in neighbours:
            neighbourLocation = tuple(neighbour)
            # dropoff += ((ndArray[location] - ndArray[neighbourLocation]) ** 2) / ndArray[location]
            dropoff += ((self.chunk_array[location] - self.chunk_array[neighbourLocation]) ** 2)
        if dropoff > 0:
            dropoff = dropoff / self.chunk_array[location]
            return math.sqrt(dropoff / len(neighbours))
        return 0


    def get_strength(self, clusterCenter, questionPoint):
        dist = euclidean_point_distance(clusterCenter, questionPoint)
        # strength = ndArray[expansionPoint] / ndArray[questionPoint] / dist

        # TODO
        strength = self.chunk_array[questionPoint] / dist / self.chunk_array[clusterCenter]

        return strength


    def expand_all_cluster_centers(self):
        for id, cc in enumerate(self.cluster_centers):
            if self.labels_array[cc] != 0:
                continue  # cluster was already discovered
            self.expand_cluster_center(cc, id + 1)

        # bring cluster labels back to (-1) - ("nr of clusters"-2) range
        uniqueClusterLabels = np.unique(self.labels_array)
        nrClust = len(uniqueClusterLabels)
        for label in range(len(uniqueClusterLabels)):
            if uniqueClusterLabels[label] == -1 or uniqueClusterLabels[label] == 0:  # don`t remark noise/ conflicta
                nrClust -= 1
                continue

    def expand_cluster_center(self, start, currentLabel):  # TODO
        """
        Expansion
        :param start: tuple - the coordinates of the chunk where the expansion starts (current cluster center)
        :param currentLabel: integer - the label of the current cluster center

        :returns labels: matrix - updated matrix of labels after expansion and conflict solve
        """
        visited = np.zeros_like(self.chunk_array, dtype=bool)
        expansionQueue = []
        if self.labels_array[start] == 0:
            expansionQueue.append(start)
            self.labels_array[start] = currentLabel

        visited[start] = True

        dropoff = self.get_dropoff(start)

        while expansionQueue:
            point = expansionQueue.pop(0)
            neighbours = get_valid_neighbours(point, np.shape(self.chunk_array))

            for neighbour in neighbours:
                location = tuple(neighbour)
                if self.version == 1:
                    number = dropoff * math.sqrt(euclidean_point_distance(start, location))
                elif self.version == 2:
                    number = math.floor(math.sqrt(dropoff * euclidean_point_distance(start, location)))

                if self.chunk_array[location] == 0:
                    pass
                if (not visited[location]) and (number < self.chunk_array[location] <= self.chunk_array[point]):
                    visited[location] = True
                    if self.labels_array[location] == currentLabel:
                        expansionQueue.append(location)
                    elif self.labels_array[location] == 0:
                        expansionQueue.append(location)
                        self.labels_array[location] = currentLabel
                    else:
                        if self.version == 0:
                            self.labels_array[location] = -1
                        else:
                            oldLabel = self.labels_array[location]
                            disRez = self.disambiguate(location,
                                                  self.cluster_centers[currentLabel - 1],
                                                  self.cluster_centers[oldLabel - 1],
                                                  )
                            # print(currentLabel, oldLabel, disRez)
                            if disRez == 1:
                                self.labels_array[location] = currentLabel
                                expansionQueue.append(location)
                            elif disRez == 2 and self.version == 2:
                                self.labels_array[location] = oldLabel
                                expansionQueue.append(location)
                            elif disRez == 11:
                                # current label wins
                                self.labels_array[self.labels_array == oldLabel] = currentLabel
                                expansionQueue.append(location)
                            elif disRez == 22:
                                # old label wins
                                self.labels_array[self.labels_array == currentLabel] = oldLabel
                                currentLabel = oldLabel
                                expansionQueue.append(location)



    def disambiguate(self, questionPoint, clusterCenter1, clusterCenter2):
        """
        Disambiguation of the cluster of a chunk based on the parameters
        :param questionPoint: tuple - the coordinates of the chunk toward which the expansion is going
        :param clusterCenter1: tuple - the coordinates of the chunk of the first cluster center
        :param clusterCenter2: tuple - the coordinates of the chunk of the second cluster center

        :returns : integer - representing the approach to disambiguation
        """
        # CHOOSE CLUSTER FOR ALREADY ASSIGNED POINT
        # usually wont get to this
        if (clusterCenter1 == questionPoint) or (clusterCenter2 == questionPoint):
            # here the point in question as already been assigned to one cluster, but another is trying to accumulate it
            # we check which of the 2 has a count and merged them
            if self.chunk_array[clusterCenter1] > self.chunk_array[clusterCenter2]:
                return 11
            else:
                return 22

        # MERGE
        # cluster 2 was expanded first, but it is actually connected to a bigger cluster
        if self.chunk_array[clusterCenter2] == self.chunk_array[questionPoint]:
            return 11
        if self.version == 2:
            # cluster 1 was expanded first, but it is actually connected to a bigger cluster
            if self.chunk_array[clusterCenter1] == self.chunk_array[questionPoint]:
                return 22

        # XANNY
        if self.version == 1:
            distanceToC1 = euclidean_point_distance(questionPoint, clusterCenter1)
            distanceToC2 = euclidean_point_distance(questionPoint, clusterCenter2)
            pointStrength = self.chunk_array[questionPoint]

            c1Strength = self.chunk_array[clusterCenter1] / pointStrength - self.get_dropoff(clusterCenter1) * distanceToC1
            c2Strength = self.chunk_array[clusterCenter2] / pointStrength - self.get_dropoff(clusterCenter2) * distanceToC2

        # RICI
        elif self.version == 2:
            c1Strength = self.get_strength(clusterCenter1, questionPoint)
            c2Strength = self.get_strength(clusterCenter2, questionPoint)

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



    def chunkify_numpy(self):

        # floor is done before the iteration over a for because it will be faster using numpy
        R = np.floor(self.chunked_data).astype(int)

        # represents the if from before, instead applied on the whole nd-array
        # (this way removing outliers, points that contain pn <- unavoidable by normalization)
        R = R[np.all(R < self.pn, axis=1)]

        # TODO FutureWarning R will have to be tuple

        # adding (the counts) in the nArray using the coordinates of R
        R = np.transpose(R).tolist()
        np.add.at(self.chunk_array, R, 1)





    def dechunkify_numpy(self):
        # floor is done before the iteration over a for because it will be faster using numpy
        R = np.floor(self.chunked_data).astype(int)

        # (this way removing outliers, points that contain pn <- unavoidable by normalization)
        R[R >= self.pn] = self.pn - 1

        # TODO FutureWarning R will have to be tuple

        # get the coordinates in the nArray (of all points) in Q and use that to set the labels of all the points in the dataset
        Q = np.transpose(R).tolist()
        self.labels = self.labels_array[Q]





if __name__ == "__main__":
    os.chdir("../")

    data, y = get_dataset_simulation_pca_2d(4)
    sp.plot('GT' + str(len(data)), data, y, marker='o')

    pn = 25
    sbm = SBM(data, pn, threshold=5)
    sbm.fit()
    sp.plot_grid(f'SBM(PN={pn}) on Sim4', data, pn, sbm.labels, marker='o')

    plt.show()