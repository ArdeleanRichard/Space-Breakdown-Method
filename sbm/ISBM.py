import math
import os
from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import networkx as nx

from .common.distance import euclidean_point_distance
from .common.neighbourhood import get_neighbours
from .dataset_parsing.simulations_dataset import get_dataset_simulation_pca_2d
from .visualization import scatter_plot as sp
from .visualization.label_map import LABEL_COLOR_MAP


class ISBM:
    def __init__(self, data, pn=25, threshold=5, adaptive=True, version=2):
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
        self.adaptive = adaptive
        self.version = version

        self.nrDim = self.data.shape[-1]

        self.graph = None
        self.cluster_centers = []
        self.labels = None


    def fit(self):
        """
        ISBM execution function

        The five steps of ISBM:
        (1) Data Normalization
        (2) Graph creation by Chunkification
        (3) Centroid Search
        (4) Expansion of Centroids
        (5) Dechunkification

        """
        # (1) Data Normalization
        self.data_preprocessing()

        # (2) Graph creation by Chunkification
        self.create_graph()

        # (3) Centroid Search
        self.get_cluster_centers()

        # (4) Expansion of Centroids
        self.expand_all_cluster_centers()

        # (5) Dechunkification
        self.get_labels()



    def data_preprocessing(self):
        """
        (1) Data Normalization - First Step of ISBM
        Min-Max scaling of the input, returning values within the [0, PN] interval
        :param X: matrix - the points of the dataset
        :param pn: int - the partioning number parameter
        :param adaptivePN: boolean - activation of the second improvement of ISBM with relation to the number of partitioning number per dimension

        :returns X_std: matrix - scaled dataset

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        """
        if self.adaptive == True:
            self.data = preprocessing.MinMaxScaler((0, 1)).fit_transform(self.data)
            feature_variance = np.var(self.data, axis=0)

            feature_variance = feature_variance / np.amax(feature_variance)
            self.pn = (feature_variance * self.pn).astype(int)

            self.data = self.data * self.pn

        else:
            self.data = preprocessing.MinMaxScaler((0, pn)).fit_transform(self.data)
            self.pn = np.full(self.pn, shape=(self.nrDim, )).astype(int)

        self.chunked_data = np.floor(self.data).astype(int)


    def create_graph(self):
        """
        (2) Graph creation by Chunkification - Second step of ISBM
        """
        self.graph = nx.Graph()

        for sample in self.chunked_data:
            string_sample = sample.tostring()
            if string_sample in self.graph:
                self.graph.nodes[string_sample]['count'] += 1
            else:
                self.graph.add_node(string_sample, count=1, label=-1)

        for node in list( self.graph.nodes):
            neighbours = get_neighbours(np.fromstring(node, dtype=int))

            for neighbour in neighbours:
                string_neighbour = neighbour.tostring()
                if string_neighbour in self.graph:
                    self.graph.add_edge(node, string_neighbour)


    def check_maxima(self, count, spike_id):
        """
        Check whether the chunk with the coordinates "point" is bigger than all its neighbours
        :param count: int - the value of the highest chunk in neighbourhood
        :param spike_id: vector - ID of the current sample

        :returns : boolean - whether or not it is a maxima
        """
        neighbours = list(self.graph.neighbors(spike_id))
        for neighbour in neighbours:
            if self.graph.nodes[neighbour]['count'] > count:
                return False
        return True


    def get_cluster_centers(self):
        """
        (3) Centroid Search - Third step of ISBM
        Search through the matrix of chunks to find the cluster centers
        :param graph: graph - the graph of chunks
        :param ccThreshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

        :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
        """

        for node in list(self.graph.nodes):
            count = self.graph.nodes[node]['count']
            if count >= self.threshold and self.check_maxima(count, node):
                self.cluster_centers.append(node)



    def get_dropoff(self, current):
        neighbours = list(self.graph.neighbors(current))
        counts = np.array([self.graph.nodes[neighbour]['count'] for neighbour in neighbours])
        dropoff = self.graph.nodes[current]['count'] - np.mean(counts)

        if dropoff > 0:
            return dropoff
        return 0

    def expand_all_cluster_centers(self):
        for id, cc in enumerate(self.cluster_centers):
            self.expand_cluster_center(cc, label=id)

    def expand_cluster_center(self, center, label):
        """
        (4) Expansion of Centroids in the graph - Fourth step of ISBM
        :param graph: graph - the graph of chunks
        :param center: node - the chunk where the expansion starts (current cluster center)
        :param label: integer - the label of the current cluster center
        :param cluster_centers: vector - vector of all the cluster centers, each containing n-dimensions

        """

        # This is done for each expansion in order to be able to disambiguate
        for node in list(self.graph.nodes):
            self.graph.nodes[node]['visited'] = 0

        expansionQueue = deque()

        if self.graph.nodes[center]['label'] == -1:
            expansionQueue.append(center)
            self.graph.nodes[center]['label'] = label

        self.graph.nodes[center]['visited'] = 1

        dropoff = self.get_dropoff(center)

        while expansionQueue:
            current = expansionQueue.popleft()

            # TODO should you pass through the connected component knowing that neighbours^3 can be bigger than the number of nodes? TO actually find the neighbours?
            neighbours = list(self.graph.neighbors(current))

            # neighbours = get_neighbours(np.fromstring(point, dtype=int))

            for neighbour in neighbours:

                if self.graph.nodes[neighbour]['visited'] == 0:
                    if self.version == 1:
                        number = math.floor(math.sqrt(dropoff * euclidean_point_distance(np.fromstring(center, dtype=int), np.fromstring(neighbour, dtype=int))))
                    elif self.version == 2:
                        number = 0.0

                    if number <= self.graph.nodes[neighbour]['count'] <= self.graph.nodes[current]['count']:
                        self.graph.nodes[neighbour]['visited'] = 1

                        if self.graph.nodes[neighbour]['label'] == label:
                            pass
                        elif self.graph.nodes[neighbour]['label'] == -1:
                            self.graph.nodes[neighbour]['label'] = label
                            expansionQueue.append(neighbour)
                        else:
                            oldLabel = self.graph.nodes[neighbour]['label']
                            disRez = self.disambiguate(
                                                  neighbour,
                                                  self.cluster_centers[label],
                                                  self.cluster_centers[oldLabel])

                            # print(label, oldLabel, disRez)
                            if disRez == 1:
                                self.graph.nodes[neighbour]['label'] = label
                                # print(np.fromstring(location, np.int))
                                expansionQueue.append(neighbour)
                            elif disRez == 2:
                                self.graph.nodes[neighbour]['label'] = oldLabel
                                expansionQueue.append(neighbour)
                            elif disRez == 11:
                                # current label wins
                                for node in list(self.graph.nodes):
                                    if self.graph.nodes[node]['label'] == oldLabel:
                                        self.graph.nodes[node]['label'] = label
                                        self.graph.nodes[node]['visited'] = 1
                                expansionQueue.append(neighbour)
                            elif disRez == 22:
                                # old label wins
                                for node in list(self.graph.nodes):
                                    if self.graph.nodes[node]['label'] == label:
                                        self.graph.nodes[node]['label'] = oldLabel
                                        self.graph.nodes[node]['visited'] = 1
                                label = oldLabel
                                expansionQueue.append(neighbour)


    def get_strength(self, clusterCenter, questionPoint):
        dist = euclidean_point_distance(np.fromstring(questionPoint, dtype=int), np.fromstring(clusterCenter, dtype=int))
        strength = self.graph.nodes[questionPoint]['count'] / dist / self.graph.nodes[clusterCenter]['count']
        return strength


    def disambiguate(self, questionPoint, current_cluster, old_cluster):
        if (current_cluster == questionPoint) or (old_cluster == questionPoint):
            if self.graph.nodes[current_cluster]['count'] > self.graph.nodes[old_cluster]['count']:
                return 11
            else:
                return 22

        # MERGE
        # cluster 2 was expanded first, but it is actually connected to a bigger cluster
        if self.graph.nodes[old_cluster]['count'] == self.graph.nodes[questionPoint]['count']:
            return 11
        # cluster 1 was expanded first, but it is actually connected to a bigger cluster
        if self.graph.nodes[current_cluster]['count'] == self.graph.nodes[questionPoint]['count']:
            return 22

        if self.version == 1:
            c1Strength = self.get_strength(current_cluster, questionPoint)
            c2Strength = self.get_strength(old_cluster, questionPoint)
        elif self.version == 2:
            distanceToC1 = euclidean_point_distance(np.fromstring(questionPoint, dtype=int), np.fromstring(current_cluster, dtype=int))
            distanceToC2 = euclidean_point_distance(np.fromstring(questionPoint, dtype=int), np.fromstring(old_cluster, dtype=int))

            c1Strength = self.graph.nodes[current_cluster]['count'] / self.get_dropoff(current_cluster) - distanceToC1
            c2Strength = self.graph.nodes[old_cluster]['count'] / self.get_dropoff(old_cluster) - distanceToC2


        if c1Strength > c2Strength:
            return 1
        else:
            return 2


    def get_labels(self):
        """
        (5) Dechunkification - Fifth and last step of ISBM
        Transforms the labels of the chunks/nodes into the labels of each of the points
        :param graph: graph - the graph of chunks
        :param spikes: matrix - contains the dataset of samples

        :returns labels: vector - the labels of the points
        """
        self.labels = []

        for sample in self.chunked_data:
            string_sample = sample.tostring()
            self.labels.append(self.graph.nodes[string_sample]['label'])

        self.labels = np.array(self.labels)


    def plot_result_grid(self, title, plot=True, marker='o'):
        """
        Plots the dataset with grid
        :param title: string - the title of the plot
        :param X: matrix - the points of the dataset
        :param pn: integer - the number of partitions on columns and rows
        :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
        :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
        :param marker: character - optional, the marker of the plot

        :returns None
        """
        if plot:
            label_color = [LABEL_COLOR_MAP[l] for l in self.labels]
            fig = plt.figure()
            plt.title(title)
            if self.nrDim == 2:
                ax = fig.gca()

                ax.set_xticks(np.arange(0, self.pn[0], 1))
                ax.set_yticks(np.arange(0, self.pn[1], 1))

                plt.scatter(self.data[:, 0], self.data[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
                plt.grid(True)
            if self.nrDim == 3:
                ax = Axes3D(fig)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")

                ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], marker=marker, c=label_color, s=25)
                # plt.grid(True)
            # fig.savefig("cevajeg.svg", format='svg', dpi=1200)


if __name__ == "__main__":
    os.chdir("../")

    data, y = get_dataset_simulation_pca_2d(4)
    sp.plot('GT' + str(len(data)), data, y, marker='o')

    pn = 25
    isbm = ISBM(data, pn, threshold=5, adaptive=True, version=2)
    isbm.fit()
    isbm.plot_result_grid(f'ISBM(PN={pn}) on Sim4', marker='o')

    plt.show()