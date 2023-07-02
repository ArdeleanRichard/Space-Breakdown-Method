import math
from collections import deque

import numpy as np
from sklearn import preprocessing
import networkx as nx

from common.distance import euclidean_point_distance
from common.neighbourhood import get_neighbours


def run(spikes, pn, ccThreshold=5, adaptivePN = False):
    """
    ISBM main function
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param ccThreshold: integer - the minimum number of points needed in a partition/chunk for it to be considered a possible cluster center
    :param adaptivePN: boolean - activation of the second improvement of ISBM with relation to the number of partitioning number per dimension

    :returns labels: vector -  the vector of labels for each point of dataset X

    The five steps of ISBM:
    (1) Data Normalization
    (2) Graph creation by Chunkification
    (3) Centroid Search
    (4) Expansion of Centroids
    (5) Dechunkification

    """
    # (1) Data Normalization
    spikes, pn = data_preprocessing(spikes, pn, adaptivePN=adaptivePN)
    spikes = np.floor(spikes).astype(int)

    # (2) Graph creation by Chunkification
    graph = create_graph(spikes)

    # (3) Centroid Search
    cluster_centers = get_cluster_centers(graph, ccThreshold)

    # (4) Expansion of Centroids
    label = 0
    for cc in cluster_centers:
        expand_cluster_center(graph, cc, label, cluster_centers)
        label += 1

    # (5) Dechunkification
    labels = get_labels(graph, spikes)

    return np.array(labels)


def data_preprocessing(spikes, pn, adaptivePN=False):
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
    if adaptivePN == True:
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        feature_variance = np.var(spikes, axis=0)

        feature_variance = feature_variance / np.amax(feature_variance)
        feature_variance = feature_variance * pn

        spikes = spikes * np.array(feature_variance)

        return spikes, feature_variance

    spikes = preprocessing.MinMaxScaler((0, pn)).fit_transform(spikes)

    return spikes, pn



def create_graph(spikes):
    """
    (2) Graph creation by Chunkification - Second step of ISBM
    :param spikes: matrix - the points of the dataset

    :returns g: graph -  the graph created through chunkification
    """
    g = nx.Graph()

    for spike in spikes:
        string_spike = spike.tostring()
        if string_spike in g:
            g.nodes[string_spike]['count'] += 1
        else:
            g.add_node(string_spike, count=1, label=-1)

    for node in list(g.nodes):
        neighbours = get_neighbours(np.fromstring(node, dtype=int))

        for neighbour in neighbours:
            string_neighbour = neighbour.tostring()
            if string_neighbour in g:
                g.add_edge(node, string_neighbour)

    return g


def check_maxima(graph, count, spike_id):
    """
    Check whether the chunk with the coordinates "point" is bigger than all its neighbours
    :param graph: graph - the graph of chunks
    :param count: int - the value of the highest chunk in neighbourhood
    :param spike_id: vector - ID of the current sample

    :returns : boolean - whether or not it is a maxima
    """
    neighbours = list(graph.neighbors(spike_id))
    for neighbour in neighbours:
        if graph.nodes[neighbour]['count'] > count:
            return False
    return True


def get_cluster_centers(graph, ccThreshold):
    """
    (3) Centroid Search - Third step of ISBM
    Search through the matrix of chunks to find the cluster centers
    :param graph: graph - the graph of chunks
    :param ccThreshold: integer - cluster center threshold, minimum amount needed for a chunk to be considered a possible cluster center

    :returns clusterCenters: vector - a vector of the coordinates of the chunks that are cluster centers
    """
    centers = []
    for node in list(graph.nodes):
        count = graph.nodes[node]['count']
        if count >= ccThreshold and check_maxima(graph, count, node):
            centers.append(node)

    return centers


def get_dropoff(graph, current):
    neighbours = list(graph.neighbors(current))
    counts = np.array([graph.nodes[neighbour]['count'] for neighbour in neighbours])
    dropoff = graph.nodes[current]['count'] - np.mean(counts)

    if dropoff > 0:
        return dropoff
    return 0



def expand_cluster_center(graph, center, label, cluster_centers):
    """
    (4) Expansion of Centroids in the graph - Fourth step of ISBM
    :param graph: graph - the graph of chunks
    :param center: node - the chunk where the expansion starts (current cluster center)
    :param label: integer - the label of the current cluster center
    :param cluster_centers: vector - vector of all the cluster centers, each containing n-dimensions

    """

    # This is done for each expansion in order to be able to disambiguate
    for node in list(graph.nodes):
        graph.nodes[node]['visited'] = 0

    expansionQueue = deque()

    if graph.nodes[center]['label'] == -1:
        expansionQueue.append(center)
        graph.nodes[center]['label'] = label

    graph.nodes[center]['visited'] = 1

    dropoff = get_dropoff(graph, center)

    while expansionQueue:
        current = expansionQueue.popleft()

        #TODO should you pass through the connected component knowing that neighbours^3 can be bigger than the number of nodes? TO actually find the neighbours?
        neighbours = list(graph.neighbors(current))

        #neighbours = get_neighbours(np.fromstring(point, dtype=int))

        for neighbour in neighbours:

            if graph.nodes[neighbour]['visited'] == 0:
                if graph.nodes[neighbour]['count'] <= graph.nodes[current]['count']:
                    graph.nodes[neighbour]['visited'] = 1

                    if graph.nodes[neighbour]['label'] == label:
                        pass
                    elif graph.nodes[neighbour]['label'] == -1:
                        graph.nodes[neighbour]['label'] = label
                        expansionQueue.append(neighbour)
                    else:
                        oldLabel = graph.nodes[neighbour]['label']
                        disRez = disambiguate(graph,
                                              neighbour,
                                              cluster_centers[label],
                                              cluster_centers[oldLabel])

                        # print(label, oldLabel, disRez)
                        if disRez == 1:
                            graph.nodes[neighbour]['label'] = label
                            # print(np.fromstring(location, np.int))
                            expansionQueue.append(neighbour)
                        elif disRez == 2:
                            graph.nodes[neighbour]['label'] = oldLabel
                            expansionQueue.append(neighbour)
                        elif disRez == 11:
                            # current label wins
                            for node in list(graph.nodes):
                                if graph.nodes[node]['label'] == oldLabel:
                                    graph.nodes[node]['label'] = label
                                    graph.nodes[node]['visited'] = 1
                            expansionQueue.append(neighbour)
                        elif disRez == 22:
                            # old label wins
                            for node in list(graph.nodes):
                                if graph.nodes[node]['label'] == label:
                                    graph.nodes[node]['label'] = oldLabel
                                    graph.nodes[node]['visited'] = 1
                            label = oldLabel
                            expansionQueue.append(neighbour)


def disambiguate(graph, questionPoint, current_cluster, old_cluster):
    if (current_cluster == questionPoint) or (old_cluster == questionPoint):
        if graph.nodes[current_cluster]['count'] > graph.nodes[old_cluster]['count']:
            return 11
        else:
            return 22

    # MERGE
    # cluster 2 was expanded first, but it is actually connected to a bigger cluster
    if graph.nodes[old_cluster]['count'] == graph.nodes[questionPoint]['count']:
        return 11
    # cluster 1 was expanded first, but it is actually connected to a bigger cluster
    if graph.nodes[current_cluster]['count'] == graph.nodes[questionPoint]['count']:
        return 22

    distanceToC1 = euclidean_point_distance(np.fromstring(questionPoint, dtype=int), np.fromstring(current_cluster, dtype=int))
    distanceToC2 = euclidean_point_distance(np.fromstring(questionPoint, dtype=int), np.fromstring(old_cluster, dtype=int))

    c1Strength = graph.nodes[current_cluster]['count'] / get_dropoff(graph, current_cluster) - distanceToC1
    c2Strength = graph.nodes[old_cluster]['count'] / get_dropoff(graph, old_cluster) - distanceToC2

    if c1Strength > c2Strength:
        return 1
    else:
        return 2


def get_labels(graph, spikes):
    """
    (5) Dechunkification - Fifth and last step of ISBM
    Transforms the labels of the chunks/nodes into the labels of each of the points
    :param graph: graph - the graph of chunks
    :param spikes: matrix - contains the dataset of samples

    :returns labels: vector - the labels of the points
    """
    labels = []

    for spike in spikes:
        string_spike = spike.tostring()
        labels.append(graph.nodes[string_spike]['label'])

    return labels



