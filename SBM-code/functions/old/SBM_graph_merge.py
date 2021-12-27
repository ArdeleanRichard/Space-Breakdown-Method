import math
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

import networkx as nx


def SBM(spikes, pn, ccThreshold=5, version=2):
    spikes = preprocessing.MinMaxScaler((0, pn)).fit_transform(spikes)
    spikes = np.floor(spikes).astype(int)
    # spikes = spikes[np.all(spikes < pn, axis=1)]

    start = time.time()
    # daca sunt mai multe chunkuri decat eventuri facem merge
    graph = create_graph(spikes)
    print(f"Create Graph: {time.time() - start}")


    start = time.time()
    cluster_centers = get_cluster_centers(graph, ccThreshold)
    print(f"Search Cluster Center: {time.time() - start} with {len(cluster_centers)} centers")

    # ccs=[]
    # for cc in cluster_centers:
    #     # print(np.fromstring(cc, dtype=int))
    #     ccs.append(np.fromstring(cc, dtype=int))
    #
    # ccs = np.array(ccs)
    # ccs = ccs[np.lexsort(ccs[:, ::-1].T)]
    #
    # cluster_centers = []
    # for cc in ccs:
    #     cluster_centers.append(cc.tostring())

    start = time.time()
    label = 1
    for cc in cluster_centers:
        expand_cluster_center(graph, cc, label, cluster_centers, version)
        label += 1
    print(f"Expansion: {time.time() - start}")

    start = time.time()
    labels = get_labels(graph, spikes, pn, merge_nodes=False)
    print(f"Labeling: {time.time() - start}")

    while count_removed(graph)/len(list(graph.nodes))*100 > 90:
    # TEST VERSION
    # while (len(labels)-np.count_nonzero(labels))/len(labels)*100 > 10:
        merge_nodes(graph, len(spikes[0]))

        start = time.time()
        cluster_centers = get_cluster_centers(graph, ccThreshold)
        print(f"Search Cluster Center: {time.time() - start} with {len(cluster_centers)} centers")

        start = time.time()
        label = 1
        for cc in cluster_centers:
            expand_cluster_center(graph, cc, label, cluster_centers, version)
            label += 1
        print(f"Expansion: {time.time() - start}")

        start = time.time()
        labels = get_labels(graph, spikes, pn, merge_nodes=True)
        print(f"Labeling: {time.time() - start} with {(len(labels)-np.count_nonzero(labels))/len(labels)*100}")


    return labels

def count_removed(graph):
    count = 0
    for node in list(graph.nodes):
        if graph.nodes[node]['to_remove'] != 0:
            count += 1
    return count

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


def merge_nodes(g, dimensions):
    print(len(list(g.nodes)))

    centers = get_cluster_centers(g, 5)
    for node in centers:
        # numarul de dimensiuni ** (-1,0,1) - 1
        # dupa primul pas de merge are mai multe
        # if len(list(g.neighbors(node))) >= dimensions**3-1:
        if g.nodes[node]['to_remove'] == 0 and check_maxima(g, g.nodes[node]['count'], node):
            for neighbour in list(g.neighbors(node)):
                if g.nodes[neighbour]['merged'] == 0 and g.nodes[neighbour]['to_remove'] == 0:
                    for neighbour_of_neighbour in list(g.neighbors(neighbour)):
                        if g.has_edge(node, neighbour_of_neighbour) or neighbour_of_neighbour == node:
                            pass
                        else:
                            g.add_edge(node, neighbour_of_neighbour)
                    g.nodes[node]['count'] += g.nodes[neighbour]['count']
                    g.nodes[neighbour]['to_remove'] = node
    # TEST VERSION
            # g.nodes[node]['merged'] += 1

    # for node in list(g.nodes):
    #     if g.nodes[node]['to_remove'] == 1:
    #         g.remove_node(node)
    # print(len(list(g.nodes)))

def create_graph(spikes):
    g = nx.Graph()

    index = 1
    for spike in spikes:
        string_spike = spike.tostring()
        if string_spike in g:
            g.nodes[string_spike]['count'] += 1
        else:
            g.add_node(string_spike, count=1, label=0, merged=0, to_remove=0, index=index)
            index += 1

        # for neighbour in neighbours:
        #     string_neighbour = neighbour.tostring()
        #     if string_neighbour in g:
        #         g.add_edge(string_spike, string_neighbour)

    # you dont need to do that for every spike, only for each node
    for node in list(g.nodes):
        neighbours = get_neighbours(np.fromstring(node, dtype=int))
        for neighbour in neighbours:
            string_neighbour = neighbour.tostring()
            if string_neighbour in g:
                g.add_edge(node, string_neighbour)



    # nodes = list(g.nodes)
    # for string_node1 in nodes:
    #     node1 = np.fromstring(string_node1, dtype=int)
    #     for string_node2 in nodes:
    #         node2 = np.fromstring(string_node2, dtype=int)
    #         test_neighbour = np.sum(node1-node2)
    #         if test_neighbour == 1 or test_neighbour == -1:
    #             g.add_edge(string_node1, string_node2)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # pos = nx.spring_layout(g, iterations=1000)
    # nx.draw(g, pos=pos, node_color='r', edge_color='b', node_size=250)
    # node_labels = nx.get_node_attributes(g, 'count')
    # nx.draw_networkx_labels(g, pos, labels=node_labels)
    # plt.show()

    return g


def check_maxima(graph, count, spike_id):
    # for neighbour in graph.neighbors(spike_id):
    #     if graph.nodes[neighbour]['count'] > count:
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

    # for neighbour in graph.neighbors(location):
    #     dropoff += ((graph.nodes[location]['count'] - graph.nodes[neighbour]['count']) ** 2) / graph.nodes[location]['count']

    neighbours = get_neighbours(np.fromstring(location, dtype=int))
    for neighbour in neighbours:
        string_neighbour = neighbour.tostring()
        if string_neighbour in graph:
            dropoff += ((graph.nodes[location]['count'] - graph.nodes[string_neighbour]['count']) ** 2) / graph.nodes[location]['count']
    if dropoff > 0:
        return math.sqrt(dropoff / len(set(graph.neighbors(location))))
    # for neighbour in graph.neighbors(location):
    #     dropoff += ((graph.nodes[location]['count'] - graph.nodes[neighbour]['count']) ** 2) / graph.nodes[location]['count']
    # if dropoff > 0:
    #     return math.sqrt(dropoff / len(set(graph.neighbors(location))))
    return 0


def get_distance(graph, start, point):
    difference = np.subtract(np.fromstring(start, dtype=int), np.fromstring(point, dtype=int))
    squared = np.square(difference)
    dist = math.sqrt(np.sum(squared))

    return dist
    # return len(nx.shortest_path(graph, start, point))


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
        # for location in graph.neighbors(point):
            # graph.nodes[neighbour]['count'] - prepare for dropoff
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
    # strength = ndArray[expansionPoint] / ndArray[questionPoint] / dist

    # TODO
    strength = graph.nodes[questionPoint]['count'] / dist / graph.nodes[cc]['count']

    return strength


def disambiguate(graph, questionPoint, expansionPoint, cc1, cc2, version):
    # CHOOSE CLUSTER FOR ALREADY ASSIGNED POINT
    # usually wont get to this
    if (cc1 == questionPoint) or (cc2 == questionPoint):
        # here the point in question as already been assigned to one cluster, but another is trying to accumulate it
        # we check which of the 2 has a count and merged them
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


def remake_nodes(graph, node, nr_merges):
    if nr_merges > 0:
        neighbours = get_neighbours(np.fromstring(node, dtype=int))
        for neighbour in neighbours:
            string_neighbour = neighbour.tostring()
            if not string_neighbour in graph:
                graph.add_node(string_neighbour, label=graph.nodes[node]['label'])
            remake_nodes(graph, string_neighbour, nr_merges-1)

def find_father_label(graph, node):
    if graph.nodes[node]['to_remove'] == 0:
        return graph.nodes[node]['label']
    else:
        find_father_label(graph, graph.nodes[node]['to_remove'])

def get_labels(graph, spikes, pn, merge_nodes=False):
    # spikes[spikes >= pn] = pn - 1
    labels = []

    # TEST VERSION
    # import copy
    # remade_graph = copy.deepcopy(graph)
    # remade_graph = graph.copy()
    # if merge_nodes == True:
    #     for node in list(remade_graph.nodes):
    #         nr_merges = remade_graph.nodes[node]['merged']
    #         remake_nodes(remade_graph, node, nr_merges)
    #     print(len(list(remade_graph.nodes)))
    #
    # for spike in spikes:
    #     string_spike = spike.tostring()
    #     labels.append(remade_graph.nodes[string_spike]['label'])

    if merge_nodes == True:
        labels_dict = {}
        for node in list(graph.nodes):
            if graph.nodes[node]['to_remove'] == 0:
                labels_dict[graph.nodes[node]['index']] = graph.nodes[node]['label']

        for spike in spikes:
            string_spike = spike.tostring()
            if graph.nodes[string_spike]['to_remove'] == 0:
                labels.append(graph.nodes[string_spike]['label'])
            else:
                labels.append(find_father_label(graph, string_spike))
    else:
        for spike in spikes:
            string_spike = spike.tostring()
            labels.append(graph.nodes[string_spike]['label'])

    return labels
