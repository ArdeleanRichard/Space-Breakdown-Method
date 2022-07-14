import time

import numpy as np
from sklearn import datasets, preprocessing

from functions import SBM
from functions import SBM_graph
import functions.dataset as ds
import functions.scatter_plot as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from functions import simulations_dataset as sds

def run_sbm():
    # data, y = ds.load_real_data()
    data, y = ds.load_synthetic_data(3)
    # data, y = ds.generate_star_data()
    # data, y = ds.generate_star_data2()
    # data, y = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
    # data, y = datasets.make_moons(n_samples=2000, noise=0.05)

    pn=25

    labels = SBM.best(data, pn, ccThreshold=10, version=2)


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
    labels = SBM_graph.SBM(data, pn, ccThreshold=5)
    print(f"Time: {time.time() - start : .3f}")


def run_sbm_graph():
    # # data, y = ds.load_real_data()
    # data, y = ds.load_synthetic_data(3)
    # pn = 25
    #
    # labels1 = SBM_graph.SBM(data, pn, ccThreshold=5, adaptivePN=True)
    # # labels2 = SBM_graph.SBM(data, pn, ccThreshold=5, version=2, adaptivePN=True)
    # sp.plot('GT' + str(len(data)), data, y, marker='o')
    # sp.plot_grid('SBM1-' + str(len(data)), data, pn, labels1, marker='o', adaptivePN=True)
    # # sp.plot_grid('SBM2-' + str(len(data)), data, pn, labels2, marker='o', adaptivePN=True)

    # # X, y = sds.get_dataset_simulation_pca_2d(22)
    # data, y = sds.get_dataset_simulation(22, 79, True)
    # pca_2d = PCA(n_components=2)
    # data = pca_2d.fit_transform(data)
    # pn = 46
    #
    # labels = SBM_graph.SBM(data, pn, ccThreshold=5, adaptivePN=True)
    # sp.plot('GT' + str(len(data)), data, y, marker='o')
    # sp.plot_grid('SBM-' + str(len(data)), data, pn, labels, marker='o', adaptivePN=True)


    data, y = sds.get_dataset_simulation_pca_2d(4)
    pn = 10
    labels = SBM_graph.SBM(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=10) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

    pn = 25
    labels = SBM_graph.SBM(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=25) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

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

    for (i, pn) in list([(4, 25), (6, 40), (17, 15), (26, 30)]):
        print(i)
        data = units_in_channel[i-1]
        data = np.array(data)
        pca_2d = PCA(n_components=2)
        X = pca_2d.fit_transform(data)
        km_labels = labels[i-1]

        sp.plot('Synthetic dataset (Sim1) ground truth', X, km_labels, marker='o')

        sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, adaptivePN=True)
        sp.plot_grid(f'ISBM on Channel {i}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)

    plt.show()


if __name__ == '__main__':
    # run_sbm()
    # run_sbm_graph()
    # sbm_times()
    run_real_data()
    # article_chunkify_presentation()

