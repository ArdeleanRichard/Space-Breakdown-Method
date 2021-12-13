import time

import numpy as np
from functions import SBM
from functions import SBM_graph
import functions.dataset as ds
import functions.scatter_plot as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_sbm():
    # data, y = ds.load_real_data()
    data, y = ds.load_synthetic_data(3)

    pn=25

    labels = SBM.best(data, pn, ccThreshold=5, version=2)


    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')

    plt.show()


def sbm_times():
    data, y = ds.load_real_data()
    # data, y = ds.load_synthetic_data(3)

    pn=25

    start = time.time()
    labels = SBM.sequential(data, pn, ccThreshold=5, version=2)
    print(f"Time: {time.time() - start : .3f}")

    start = time.time()
    labels = SBM.parallel(data, pn, ccThreshold=5, version=2)
    print(f"Time: {time.time() - start : .3f}")

    start = time.time()
    labels = SBM.best(data, pn, ccThreshold=5, version=2)
    print(f"Time: {time.time() - start : .3f}")

    start = time.time()
    labels = SBM_graph.SBM(data, pn, ccThreshold=5, version=2)
    print(f"Time: {time.time() - start : .3f}")


def run_sbm_graph():
    # data, y = ds.load_real_data()
    data, y = ds.load_synthetic_data(3)
    pn = 25

    labels = SBM_graph.SBM(data, pn, ccThreshold=5, version=2)

    sp.plot('GT' + str(len(data)), data, labels, marker='o')
    sp.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')

    plt.show()


def article_chunkify_presentation():
    avgPoints = 10
    C1 = [-2, 0] + .5 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .1 * np.random.randn(avgPoints * 1, 2)

    C3 = [1, -2] + .5 * np.random.randn(avgPoints * 2, 2)


    X = np.vstack((C1, C3, C4))

    c1Labels = np.full(len(C1), 1)
    c3Labels = np.full(len(C3), 3)
    c4Labels = np.full(len(C4), 4)

    y = np.hstack((c1Labels, c3Labels, c4Labels))

    from sklearn import preprocessing
    pn=5
    Xm = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)


    sp.plot_grid('50 generated points - chunkification', Xm, pn=5, labels=y, marker='o')


    from functions import SBM_functions as fs
    Xm = np.floor(Xm).astype(int)
    Xm[Xm==pn]=pn-1

    ndArray = fs.chunkify_numpy(Xm, pn)
    ndArray = fs.rotateMatrix(ndArray)
    print(ndArray)

    import networkx as nx
    graph = SBM_graph.create_graph(Xm)
    plt.figure()
    counts = {}
    for node in list(graph.nodes):
        counts[node] = graph.nodes[node]['count']

    nx.draw(graph, labels=counts, with_labels=True)
    plt.show()


def run_real_data():
    units_in_channel, labels = ds.get_M045_009()

    for (i, pn) in list([(4, 25), (6, 40), (17, 20), (26, 30)]):
        print(i)
        data = units_in_channel[i-1]
        data = np.array(data)
        pca_2d = PCA(n_components=2)
        X = pca_2d.fit_transform(data)
        km_labels = labels[i-1]

        # sp.plot('Synthetic dataset (Sim1) ground truth', X, km_labels, marker='o', alpha=0.5)

        sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2, adaptivePN=True)
        sp.plot_grid('SBM graph2', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)

    plt.show()



if __name__ == '__main__':
    # run_sbm()
    # run_sbm_graph()
    # sbm_times()
    run_real_data()
    # article_chunkify_presentation()

