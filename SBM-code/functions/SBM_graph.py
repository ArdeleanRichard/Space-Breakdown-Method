import math
import time
import numpy as np
from sklearn import preprocessing
import networkx as nx


def SBM(spikes, pn, ccThreshold=5, version=2, adaptivePN = False):
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
    label = 1
    for cc in cluster_centers:
        expand_cluster_center(graph, cc, label, cluster_centers, version)
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
            g.add_node(string_spike, count=1, label=0)

    for node in list(g.nodes):
        neighbours = get_neighbours(np.fromstring(node, dtype=int))
        for neighbour in neighbours:
            string_neighbour = neighbour.tostring()
            if string_neighbour in g:
                g.add_edge(node, string_neighbour)
    return g


def check_maxima(graph, count, spike_id):
    neighbours = get_neighbours(np.fromstring(spike_id, dtype=int))
    for neighbour in neighbours:
        string_neighbour = neighbour.tostring()
        if string_neighbour in graph and graph.nodes[string_neighbour]['count'] > count:
            return False
    return True


def get_cluster_centers(graph, ccThreshold):
    centers = []
    for node in list(graph.nodes):
        count = graph.nodes[node]['count']
        if count >= ccThreshold and check_maxima(graph, count, node):
            centers.append(node)

    return centers


def get_dropoff(graph, location):
    dropoff = 0

    neighbours = get_neighbours(np.fromstring(location, dtype=int))
    for neighbour in neighbours:
        string_neighbour = neighbour.tostring()
        if string_neighbour in graph:
            dropoff += ((graph.nodes[location]['count'] - graph.nodes[string_neighbour]['count']) ** 2) / graph.nodes[location]['count']
    if dropoff > 0:
        return math.sqrt(dropoff / len(set(graph.neighbors(location))))
    return 0


def get_distance(graph, start, point):
    difference = np.subtract(np.fromstring(start, dtype=int), np.fromstring(point, dtype=int))
    squared = np.square(difference)
    dist = math.sqrt(np.sum(squared))

    return dist


def expand_cluster_center(graph, start, label, cluster_centers, version):
    for node in list(graph.nodes):
        graph.nodes[node]['visited'] = 0

    expansionQueue = []

    if graph.nodes[start]['label'] == 0:
        expansionQueue.append(start)
        graph.nodes[start]['label'] = label

    graph.nodes[start]['visited'] = 1

    dropoff = get_dropoff(graph, start)

    while expansionQueue:
        point = expansionQueue.pop(0)

        neighbours = get_neighbours(np.fromstring(point, dtype=int))
        for neighbour in neighbours:
            location = neighbour.tostring()

            if version == 1:
                number = dropoff * math.sqrt(get_distance(graph, start, location))
            elif version == 2:
                number = math.floor(math.sqrt(dropoff * get_distance(graph, start, location)))

            try:
                if not graph.nodes[location]['visited'] and number < graph.nodes[location]['count'] <= graph.nodes[point]['count']:
                    graph.nodes[location]['visited'] = 1

                    if graph.nodes[location]['label'] == label:
                        expansionQueue.append(location)
                    elif graph.nodes[location]['label'] == 0:
                        expansionQueue.append(location)
                        graph.nodes[location]['label'] = label

                    else:
                        oldLabel = graph.nodes[location]['label']
                        disRez = disambiguate(graph,
                                              location,
                                              point,
                                              cluster_centers[label - 1],
                                              cluster_centers[oldLabel - 1],
                                              version)
                        # print(label, oldLabel, disRez)
                        if disRez == 1:
                            graph.nodes[location]['label'] = label
                            expansionQueue.append(location)
                        elif disRez == 2 and version == 2:
                            graph.nodes[location]['label'] = oldLabel
                            expansionQueue.append(location)
                        elif disRez == 11:
                            # current label wins
                            for node in list(graph.nodes):
                                if graph.nodes[node]['label'] == oldLabel:
                                    graph.nodes[node]['label'] = label
                            expansionQueue.append(location)
                        elif disRez == 22:
                            # old label wins
                            for node in list(graph.nodes):
                                if graph.nodes[node]['label'] == label:
                                    graph.nodes[node]['label'] = oldLabel
                            label = oldLabel
                            expansionQueue.append(location)


            except KeyError:
                pass


def get_strength(graph, cc, questionPoint):
    dist = get_distance(graph, cc, questionPoint)

    strength = graph.nodes[questionPoint]['count'] / dist / graph.nodes[cc]['count']

    return strength


def disambiguate(graph, questionPoint, expansionPoint, cc1, cc2, version):
    if (cc1 == questionPoint) or (cc2 == questionPoint):
        if graph.nodes[cc1]['count'] > graph.nodes[cc2]['count']:
            return 11
        else:
            return 22

    # MERGE
    # cluster 2 was expanded first, but it is actually connected to a bigger cluster
    if graph.nodes[cc2]['count'] == graph.nodes[questionPoint]['count']:
        return 11
    if version == 2:
        # cluster 1 was expanded first, but it is actually connected to a bigger cluster
        if graph.nodes[cc1]['count'] == graph.nodes[questionPoint]['count']:
            return 22

    if version == 1:
        distanceToC1 = get_distance(graph, questionPoint, cc1)
        distanceToC2 = get_distance(graph, questionPoint, cc2)
        pointStrength = graph.nodes[questionPoint]['count']

        c1Strength = graph.nodes[cc1]['count'] / pointStrength - get_dropoff(graph, cc1) * distanceToC1
        c2Strength = graph.nodes[cc2]['count'] / pointStrength - get_dropoff(graph, cc2) * distanceToC2

    elif version == 2:
        c1Strength = get_strength(graph, cc1, questionPoint)
        c2Strength = get_strength(graph, cc2, questionPoint)

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
