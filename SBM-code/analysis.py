import numpy as np
import time
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, v_measure_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from functions import SBM
from functions import SBM_graph
import functions.dataset as ds
import functions.simulations_dataset as sds
import functions.scatter_plot as sp
from metric import ss_metric, ss_metric_unweighted, ss_metric_unweighted2, ss_metric_trial2, ss_metric_trial1


def try_metric(X, y, n_clusters, eps, pn=25, version=2, no_noise=True):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5, version=1)

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2)

    sbm_graph2_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=version, adaptivePN=True)

    noise_label = [False, False, False]
    if no_noise == True:
        noise_label = [-1, 0, 0]

    print(f"KMeans: {ss_metric(y, kmeans.labels_, False):.3f}")
    print(f"DBSCAN: {ss_metric(y, dbscan.labels_, noise_label[0]):.3f}")
    print(f"SBMog: {ss_metric(y, sbm_array_labels, noise_label[1]):.3f}")
    # print(f"ISBM: {ss_metric(y, sbm_graph_labels):.3f}")
    print(f"ISBM2: {ss_metric(y, sbm_graph2_labels, noise_label[2]):.3f}")
    # test_labels = np.array(list(range(0, len(y))))
    # print(f"Test: {ss_metric(y, test_labels):.3f}")
    print()
    #
    # print()
    # print(f"KMeans: {ss_metric_unweighted(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_unweighted(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_unweighted(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_unweighted(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_unweighted(y, test_labels):.3f}")
    #
    # print()
    # print(f"KMeans: {ss_metric_unweighted2(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_unweighted2(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_unweighted2(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_unweighted2(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_unweighted2(y, test_labels):.3f}")
    # print()
    #
    # print()
    # print(f"KMeans: {ss_metric_trial1(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_trial1(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_trial1(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_trial1(y, sbm_graph2_labels, 0):.3f}")
    # # print(f"Test: {ss_metric_trial1(y, test_labels):.3f}")
    # print()
    #
    # print()
    # print(f"KMeans: {ss_metric_trial2(y, kmeans.labels_, False):.3f}")
    # print(f"DBSCAN: {ss_metric_trial2(y, dbscan.labels_, -1):.3f}")
    # print(f"SBMog: {ss_metric_trial2(y, sbm_array_labels,0):.3f}")
    # print(f"ISBM2: {ss_metric_trial2(y, sbm_graph2_labels, 0):.3f}")
    # print(f"Test: {ss_metric_trial2(y, test_labels):.3f}")
    # print()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)


def compare_result_graph_vs_array_structure(Title, X, y, n_clusters, eps, pn=25, version=2):
    sp.plot(f'Synthetic dataset ({Title})  ground truth', X, y, marker='o', alpha=0.5)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    sp.plot(f'KMeans on {Title}', X, kmeans.labels_, marker='o')

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
    sp.plot(f'DBSCAN on {Title}', X, dbscan.labels_, marker='o')

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5, version=1)
    sp.plot_grid(f'SBM array on {Title}', X, pn, sbm_array_labels, marker='o')

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=1)
    sp.plot_grid(f'SBM graph on {Title}', X, pn, sbm_graph_labels, marker='o')

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=version, adaptivePN=True)
    sp.plot_grid(f'SBM graph2 on {Title}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)

    plt.show()


def compare_time_graph_vs_array_structure(X, y, n_clusters, eps, runs=25):
    print(len(X))
    pn = 25

    kmeans_time = 0
    dbscan_time = 0
    sbm_array_time = 0
    sbm_graph_time = 0
    sbm_graph2_time = 0

    for i in range(runs):
        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        kmeans_time += (time.time() - start)

        # start = time.time()
        # dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
        # dbscan_time += (time.time() - start)

        # start = time.time()
        # sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5, version=1)
        # sbm_array_time += (time.time() - start)

        # start = time.time()
        # sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2)
        # sbm_graph_time += (time.time() - start)

        start = time.time()
        sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2, adaptivePN=True)
        sbm_graph2_time += (time.time() - start)

    print(f"Kmeans time - {runs} runs: {kmeans_time/runs:.3f}s")
    print(f"DBSCAN time - {runs} runs: {dbscan_time/runs:.3f}s")
    print(f"SBM array time - {runs} runs: {sbm_array_time/runs:.3f}s")
    print(f"SBM graph time - {runs} runs: {sbm_graph_time/runs:.3f}s")
    print(f"SBM graph2 time - {runs} runs: {sbm_graph2_time/runs:.3f}s")


def compare_metrics_graph_vs_array_structure(X, y, n_clusters, eps, pn=25):
    # dataset

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5, version=1)

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2)
    sbm_graph2_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2, adaptivePN=True)

    #metric - ARI
    print(f"UO - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_rand_score(y, dbscan.labels_):.3f}\t"
          f"SBM_array={adjusted_rand_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={adjusted_rand_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_rand_score(y, sbm_graph2_labels):.3f}\t")

    print(f"UO - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_mutual_info_score(y, dbscan.labels_):.3f}\t"
          f"SBM_array={adjusted_mutual_info_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={adjusted_mutual_info_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_mutual_info_score(y, sbm_graph2_labels):.3f}\t")

    print(f"UO - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={purity_score(y, dbscan.labels_):.3f}\t"
          f"SBM_array={purity_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={purity_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={purity_score(y, sbm_graph2_labels):.3f}\t")

    print(f"UO - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={fowlkes_mallows_score(y, dbscan.labels_):.3f}\t"
          f"SBM_array={fowlkes_mallows_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={fowlkes_mallows_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={fowlkes_mallows_score(y, sbm_graph2_labels):.3f}\t")

    print(f"UO - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={v_measure_score(y, dbscan.labels_):.3f}\t"
          f"SBM_array={v_measure_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={v_measure_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={v_measure_score(y, sbm_graph2_labels):.3f}\t")

    print(f"UO - SS: "
          f"KMeans={silhouette_score(X, kmeans.labels_):.3f}\t"
          f"DBSCAN={silhouette_score(X, dbscan.labels_):.3f}\t"
          f"SBM_array={silhouette_score(X, sbm_array_labels):.3f}\t"
          # f"SBM_graph={silhouette_score(X, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={silhouette_score(X, sbm_graph2_labels):.3f}\t")

    print(f"UO - CHS: "
          f"KMeans={calinski_harabasz_score(X, kmeans.labels_):.3f}\t"
          f"DBSCAN={calinski_harabasz_score(X, dbscan.labels_):.3f}\t"
          f"SBM_array={calinski_harabasz_score(X, sbm_array_labels):.3f}\t"
          # f"SBM_graph={calinski_harabasz_score(X, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={calinski_harabasz_score(X, sbm_graph2_labels):.3f}\t")

    print(f"UO - DBS: "
          f"KMeans={davies_bouldin_score(X, kmeans.labels_):.3f}\t"
          f"DBSCAN={davies_bouldin_score(X, dbscan.labels_):.3f}\t"
          f"SBM_array={davies_bouldin_score(X, sbm_array_labels):.3f}\t"
          # f"SBM_graph={davies_bouldin_score(X, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={davies_bouldin_score(X, sbm_graph2_labels):.3f}\t")


def compare_time_dimensions(nr_dim=3, runs=25):
    X, y = sds.get_dataset_simulation(4)
    pca_3d = PCA(n_components=nr_dim)
    X = pca_3d.fit_transform(X)

    compare_time_graph_vs_array_structure(X, y, 5, 0.1, runs)


def compare_time_samples():
    for i in range(1, 15):
        size = i * 250
        X, y = ds.generate_simulated_data(size)
        compare_time_graph_vs_array_structure(X, y, 6, 0.5, 100)


def compare_metrics_dimensions(nr_dim, n_clusters, eps, pn=25):
    X, y = sds.get_dataset_simulation(4)
    pca_nd = PCA(n_components=nr_dim)
    X = pca_nd.fit_transform(X)

    print(nr_dim)
    compare_metrics_graph_vs_array_structure(X, y, n_clusters, eps, pn)


def compare_result_dim(X, y, nr_dim, n_clusters, eps):
    pca_2d = PCA(n_components=2)
    X2D = pca_2d.fit_transform(X)

    pca_nd = PCA(n_components=nr_dim)
    X = pca_nd.fit_transform(X)

    pn = 25
    sp.plot('Synthetic dataset (UO) ground truth', X2D, y, marker='o', alpha=0.5)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    sp.plot('KMeans on UO', X2D, kmeans.labels_, marker='o')

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)
    sp.plot('DBSCAN on UO', X2D, dbscan.labels_, marker='o')

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5, version=1)
    sp.plot_grid('SBM array on UO', X2D, pn, sbm_array_labels, marker='o')

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2)
    sp.plot_grid('SBM graph on UO', X2D, pn, sbm_graph_labels, marker='o')

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5, version=2, adaptivePN=True)
    sp.plot_grid('SBM graph2 on UO', X2D, pn, sbm_graph_labels, marker='o', adaptivePN=True)

    plt.show()



##### METRIC ANALYSIS ####
no_noise=False
# X, y = ds.generate_simulated_data()
# try_metric(X, y, 6, 0.5, 25, no_noise=no_noise)
# X, y = sds.get_dataset_simulation_pca_2d(4)
# try_metric(X, y, 5, 0.1, 30, no_noise=no_noise)
# compare_result_graph_vs_array_structure(X, y, 6, 0.5)

# X, y = sds.get_dataset_simulation_pca_2d(1)
# try_metric(X, y, 17, 0.05, 46, 1, no_noise=no_noise)
# compare_result_graph_vs_array_structure('Sim1', X, y, 17, 0.1, 46, 1)

# X, y = sds.get_dataset_simulation_pca_2d(22)
# try_metric(X, y, 7, 0.05, 46, 1, no_noise=no_noise)
# compare_result_graph_vs_array_structure('Sim22', X, y, 7, 0.1, 46, 1)
# TODO 30 OG - 35 I Pn=35 1.0, 0.999
# X, y = sds.get_dataset_simulation_pca_2d(21)
# try_metric(X, y, 5, 0.1, 20, 1, no_noise=no_noise) # 20 - 98.4
# compare_result_graph_vs_array_structure('Sim21', X, y, 5, 0.1, 20, 1)
# TODO 30/35 OG - V1-Pn=40 equality but better on I
# X, y = sds.get_dataset_simulation_pca_2d(30)
# try_metric(X, y, 6, 0.1, 40, 1)
# compare_result_graph_vs_array_structure('Sim30', X, y, 6, 0.1, 40, 1)

#### BY DATASET ANALYSIS #####

# X, y = ds.generate_simulated_data()
# compare_result_graph_vs_array_structure('UO', X, y, 6, 0.5)
# compare_time_graph_vs_array_structure(X, y, 6, 0.5, 100)
# compare_metrics_graph_vs_array_structure(X, y, 6, 0.5)

# X, y = sds.get_dataset_simulation_pca_2d(30)
# compare_result_graph_vs_array_structure('Sim30', X, y, 6, 0.1, 30)
# compare_time_graph_vs_array_structure(X, y, 6, 0.5, 100)
# compare_metrics_graph_vs_array_structure(X, y, 6, 0.1, 30)

# X, y = sds.get_dataset_simulation_pca_2d(4)
# print(len(np.unique(y)))
# compare_result_graph_vs_array_structure('Sim4', X, y, 5, 0.1, 25)
# compare_time_graph_vs_array_structure(X, y, 5, 0.1, 100)
# compare_metrics_graph_vs_array_structure(X, y, 5, 0.1, 10)

# X, y = sds.get_dataset_simulation_pca_2d(21)
# print(len(np.unique(y)))
# compare_result_graph_vs_array_structure(X, y, 5, 0.1)
# compare_time_graph_vs_array_structure(X, y, 5, 0.1, 100)
# compare_metrics_graph_vs_array_structure(X, y, 5, 0.1, 35)

# X, y = sds.get_dataset_simulation_pca_2d(1)
# print(len(np.unique(y)))
# compare_result_graph_vs_array_structure(X, y, 17, 0.1)
# compare_time_graph_vs_array_structure(X, y, 17, 0.1, 100)
# compare_metrics_graph_vs_array_structure(X, y, 17, 0.05, 46)

# X, y = sds.get_dataset_simulation_pca_2d(22)
# print(len(np.unique(y)))
# compare_result_graph_vs_array_structure(X, y, 7, 0.05, 50)
# compare_time_graph_vs_array_structure(X, y, 7, 0.1, 100)
# compare_metrics_graph_vs_array_structure(X, y, 7, 0.05, 40)




#### OVERALL ANALYSIS ####
# X, y = sds.get_dataset_simulation(4)
# print(len(np.unique(y)))
# compare_time_dimensions(2, 1)
# compare_time_dimensions(3, 1)
# compare_time_dimensions(4, 1)
# compare_time_dimensions(5, 1)
# compare_time_dimensions(6, 1)
# compare_time_dimensions(7)
# compare_time_dimensions(8)
# compare_metrics_dimensions(2, 5, 0.1, 10)
# compare_metrics_dimensions(3, 5, 0.25, 12)
# compare_metrics_dimensions(4, 5, 0.4, 8)
# compare_metrics_dimensions(5, 5, 0.4)
# compare_metrics_dimensions(6, 5, 0.4)
# compare_result_dim(X, y, 2, 5, 0.1)
# compare_result_dim(X, y, 3, 5, 0.25)
# compare_result_dim(X, y, 4, 5, 0.4)

# compare_time_samples()

