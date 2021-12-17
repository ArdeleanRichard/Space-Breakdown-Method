import math
import time
import numpy as np
from sklearn import preprocessing
import networkx as nx


def SBM(spikes, pn, ccThreshold=5, adaptivePN = False):
    spikes, pn = data_preprocessing(spikes, pn, adaptivePN=adaptivePN)
    spikes = np.floor(spikes).astype(int)

    # import time
    # start = time.time()
    graph = create_graph(spikes)
    # print(f"CG: {time.time()-start}, {len(graph.nodes)}, {len(graph.edges)}")
    # print(len(graph.nodes))

    # start = time.time()
    cluster_centers = get_cluster_centers(graph, ccThreshold)
    # print(f"CC: {time.time() - start}")

    # start = time.time()
    label = 0
    for cc in cluster_centers:
        expand_cluster_center(graph, cc, label, cluster_centers)
        label += 1
    # print(f"EX: {time.time() - start}")

    # start = time.time()
    labels = get_labels(graph, spikes)
    # print(f"DL: {time.time() - start}")


    return np.array(labels)


def data_preprocessing(spikes, pn, adaptivePN=False):
    if adaptivePN == True:
        # feature_variance = np.var(spikes, axis=0)
        # print(feature_variance)

        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        feature_variance = np.var(spikes, axis=0)
        # print(feature_variance)

        # pca = PCA(n_components=2)
        # pca.fit(spikes)
        # feature_variance = pca.explained_variance_ratio_
        feature_variance = feature_variance / np.amax(feature_variance)
        feature_variance = feature_variance * pn
        # feature_variance[1] = feature_variance[1] * 3
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        spikes = spikes * np.array(feature_variance)
        # print(feature_variance)

        return spikes, feature_variance

    spikes = preprocessing.MinMaxScaler((0, pn)).fit_transform(spikes)

    return spikes, pn


def get_neighbours(point):
    # ndim = the number of dimensions of a point=chunk
    ndim = len(point)

    # offsetIndexes gives all the possible neighbours ( (0,0)...(2,2) ) of an unknown point in n-dimensions
    offsetIndexes = np.indices((3,) * ndim).reshape(ndim, -1).T

    # np.r_ does row-wise merging (basically concatenate), this instructions is equivalent to offsets=np.array([-1, 0, 1]).take(offsetIndexes)
    offsets = np.r_[-1, 0, 1].take(offsetIndexes)

    # remove the point itself (0,0) from the offsets (np.any will give False only for the point that contains only 0 on all dimensions)
    offsets = offsets[np.any(offsets, axis=1)]

    # calculate the coordinates of the neighbours of the point using the offsets
    neighbours = point + offsets

    return neighbours


def create_graph(spikes):
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

    #print(nx.connected_components(g))
    #print(len(list(nx.connected_components(g))))
    return g


def check_maxima(graph, count, spike_id):
    neighbours = list(graph.neighbors(spike_id))
    #neighbours = get_neighbours(np.fromstring(spike_id, dtype=int))
    # for neighbour in neighbours:
    #    string_neighbour = neighbour.tostring()
    #    if neighbour in graph and graph.nodes[string_neighbour]['count'] > count:
    for neighbour in neighbours:
        if graph.nodes[neighbour]['count'] > count:
            return False
    return True


def get_cluster_centers(graph, ccThreshold):
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


def get_distance(start, point):
    difference = np.subtract(np.fromstring(start, dtype=int), np.fromstring(point, dtype=int))
    squared = np.square(difference)
    dist = math.sqrt(np.sum(squared))

    return dist


def expand_cluster_center(graph, center, label, cluster_centers):
    # This is done for each expansion in order to be able to disambiguate
    for node in list(graph.nodes):
        graph.nodes[node]['visited'] = 0

    expansionQueue = []

    if graph.nodes[center]['label'] == -1:
        expansionQueue.append(center)
        graph.nodes[center]['label'] = label

    graph.nodes[center]['visited'] = 1

    dropoff = get_dropoff(graph, center)

    while expansionQueue:
        current = expansionQueue.pop(0)

        #TODO should you pass through the connected component knowing that neighbours^3 can be bigger than the number of nodes? TO actually find the neighbours?
        neighbours = list(graph.neighbors(current))

        #neighbours = get_neighbours(np.fromstring(point, dtype=int))

        for neighbour in neighbours:
        #for neighbour in neighbours:
            #location = neighbour.tostring()

            if graph.nodes[neighbour]['visited'] == 0:
                if graph.nodes[neighbour]['count'] <= graph.nodes[current]['count']:
                    graph.nodes[neighbour]['visited'] = 1

                    if graph.nodes[neighbour]['label'] == label:
                        # Arrives here when disRez 11 or 22
                        pass
                        # expansionQueue.append(location)
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


    distanceToC1 = get_distance(questionPoint, current_cluster)
    distanceToC2 = get_distance(questionPoint, old_cluster)

    c1Strength = graph.nodes[current_cluster]['count'] / get_dropoff(graph, current_cluster) - distanceToC1
    c2Strength = graph.nodes[old_cluster]['count'] / get_dropoff(graph, old_cluster) - distanceToC2

    if c1Strength > c2Strength:
        return 1
    else:
        return 2


def get_labels(graph, spikes):
    labels = []

    for spike in spikes:
        string_spike = spike.tostring()
        labels.append(graph.nodes[string_spike]['label'])

    return labels
