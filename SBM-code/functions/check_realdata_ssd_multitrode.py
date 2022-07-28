import numpy as np
from fcmeans import FCM
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, v_measure_score

from functions import SBM_graph, SBM
from functions.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, select_data, plot_multitrode, plot_multitrodes
from functions.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from functions.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel, plot_sorted_data

# DATASET_PATH = '../../data/M046_0001_MT/'
# DATASET_PATH = '../../data/M045_RF_0008_19_MT/'
# DATASET_PATH = '../../data/M045_SRCS_0009_MT/'
# DATASET_PATH = '../../data/M045_DRCT_0015_MT/'
# DATASET_PATH = '../../data/M017_0004_MT/'
# DATASET_PATH = '../../data/M017_MT/'
from functions.scores import purity_score
from metric import ss_metric

# DATASET_PATH = '../../data/M017_0004_Tetrode/Units/'
DATASET_PATH = '../../data/M017_0004_Tetrode_try2/ssd/'
# DATASET_PATH = '../../data/M017_0004_Tetrode8/'

spikes_per_unit, unit_multitrode, _ = parse_ssd_file(DATASET_PATH)
MULTITRODE_WAVEFORM_LENGTH = 232
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_MULTITRODES = 8
NR_ELECTRODES_PER_MULTITRODE = 4

print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_multitrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_multitrode}")
print("--------------------------------------------")

print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")

timestamps = read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_unit)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_unit)}")

timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_unit))}")
print(f"Assert equality: {list(spikes_per_unit) == list(map(len, timestamps_by_unit))}")
print("--------------------------------------------")


event_timestamps = read_event_timestamps(event_timestamps_filename)
print(f"Event Timestamps found in file: {event_timestamps.shape}")
event_codes = read_event_codes(event_codes_filename)
print(f"Event Codes found in file: {event_codes.shape}")
# print(event_timestamps)
# print(event_codes)
print(f"Assert equality: {list(event_timestamps) == len(event_codes)}")
print("--------------------------------------------")

waveforms = read_waveforms(waveform_file)

# Check between stimulus
# print(event_timestamps)
# print(event_codes)
# print(waveforms.shape)
# timestamp_start = event_timestamps[1]
# timestamp_stop = event_timestamps[81]
# waveforms_reshaped = waveforms.reshape((-1,58*4))
# print(timestamps.shape)
# print(waveforms_reshaped.shape)
# print((timestamps > timestamp_start).shape)
# cond = np.logical_and(timestamps > timestamp_start, timestamps < timestamp_stop)
# waveforms = waveforms_reshaped[cond]
# waveforms = waveforms.reshape((1, -1))[0]
# print(waveforms.shape)

print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*{MULTITRODE_WAVEFORM_LENGTH}: {len(timestamps) * MULTITRODE_WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * MULTITRODE_WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, MULTITRODE_WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
waveform_lens = list(map(len, waveforms_by_unit))
print(f"Waveforms/{MULTITRODE_WAVEFORM_LENGTH} per channel should be equal: {[i//MULTITRODE_WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_unit) == [i//MULTITRODE_WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")

units_in_multitrode, labels = units_by_channel(unit_multitrode, waveforms_by_unit, data_length=MULTITRODE_WAVEFORM_LENGTH, number_of_channels=NR_MULTITRODES)
# for unit in units_in_multitrode:
#     print(len(unit))
units_by_multitrodes = split_multitrode(units_in_multitrode, MULTITRODE_WAVEFORM_LENGTH, WAVEFORM_LENGTH)
# for unit in units_by_multitrodes:
#     print(len(unit))

# data = select_data(data=units_by_multitrodes, multitrode_nr=0, electrode_in_multitrode=0)
# plot_multitrodes(units_by_multitrodes, labels, nr_multitrodes=NR_MULTITRODES, nr_electrodes=NR_ELECTRODES_PER_MULTITRODE)
# plot_multitrode(units_by_multitrodes, labels, 1, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 3, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 5, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 7, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)

data = units_by_multitrodes[7][0]
labels = labels[7]
data = np.array(data)
pca_ = PCA(n_components=3)
data_pca = pca_.fit_transform(data)
import scatter_plot as sp
import matplotlib.pyplot as plt

sp.plot("Ground truth of Electrode 1", data_pca, labels)

# kmeans = KMeans(n_clusters=4, random_state=0).fit(data_pca)
# sp.plot('K-Means on Electrode 1', data_pca, kmeans.labels_, marker='o')
# dbscan = DBSCAN(eps=18, min_samples=np.log(len(data_pca))).fit(data_pca)
# sp.plot('DBSCAN on Electrode 1', data_pca, dbscan.labels_, marker='o')
sbm_array_labels = SBM.sequential(data_pca, pn=10, ccThreshold=5)
sp.plot('SBM on Electrode 1', data_pca, sbm_array_labels, marker='o')
# sbm_graph2_labels = SBM_graph.SBM(data_pca, pn=20, ccThreshold=5, adaptivePN=True)
# sp.plot('ISBM on Electrode 1', data_pca, sbm_graph2_labels, marker='o')

# bandwidth = estimate_bandwidth(data_pca, quantile=0.05, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data_pca)
# sp.plot('MeanShift on Electrode 1', data_pca, ms.labels_, marker='o')
# ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(data_pca)
# sp.plot('AgglomerativeClustering on Electrode 1', data_pca, ward.labels_, marker='o')
# my_model = FCM(n_clusters=4)
# my_model.fit(data_pca)
# labels = my_model.predict(data_pca)
# sp.plot('FCM on Electrode 1', data_pca, labels, marker='o')

plt.show()

def try_metric(X, y, n_clusters, eps, pn=25, no_noise=True):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5)

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5)

    sbm_graph2_labels = SBM_graph.SBM(X, pn, ccThreshold=5, adaptivePN=True)

    bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)

    print(f"KMeans: {ss_metric(y, kmeans.labels_):.3f}")
    print(f"DBSCAN: {ss_metric(y, dbscan.labels_):.3f}")
    print(f"MS: {ss_metric(y, ms.labels_):.3f}")
    print(f"AC: {ss_metric(y, ward.labels_):.3f}")
    print(f"FCM: {ss_metric(y, labels):.3f}")
    print(f"SBMog: {ss_metric(y, sbm_array_labels):.3f}")
    print(f"ISBM: {ss_metric(y, sbm_graph_labels):.3f}")
    print(f"ISBM: {ss_metric(y, sbm_graph2_labels):.3f}")
    # test_labels = np.array(list(range(0, len(y))))
    # print(f"Test: {ss_metric(y, test_labels):.3f}")

def compare_metrics_graph_vs_array_structure(Data, X, y, n_clusters, eps, pn=25):
    # dataset

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    dbscan = DBSCAN(eps=eps, min_samples=np.log(len(X))).fit(X)

    sbm_array_labels = SBM.sequential(X, pn, ccThreshold=5)

    sbm_graph_labels = SBM_graph.SBM(X, pn, ccThreshold=5)
    sbm_graph2_labels = SBM_graph.SBM(X, pn, ccThreshold=5, adaptivePN=True)

    bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)

    my_model = FCM(n_clusters=n_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)

    # metric - ARI
    print(f"{Data} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_rand_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_rand_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_rand_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_rand_score(y, labels):.3f}\t"
          # f"SBM_graph={adjusted_rand_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_rand_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_mutual_info_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_mutual_info_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_mutual_info_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_mutual_info_score(y, labels):.3f}\t"
          f"SBM_array={adjusted_mutual_info_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={adjusted_mutual_info_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={adjusted_mutual_info_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={purity_score(y, dbscan.labels_):.3f}\t"
          f"MS={purity_score(y, ms.labels_):.3f}\t"
          f"Ag={purity_score(y, ward.labels_):.3f}\t"
          f"FCM={purity_score(y, labels):.3f}\t"
          f"SBM_array={purity_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={purity_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={purity_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={fowlkes_mallows_score(y, dbscan.labels_):.3f}\t"
          f"MS={fowlkes_mallows_score(y, ms.labels_):.3f}\t"
          f"Ag={fowlkes_mallows_score(y, ward.labels_):.3f}\t"
          f"FCM={fowlkes_mallows_score(y, labels):.3f}\t"
          f"SBM_array={fowlkes_mallows_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={fowlkes_mallows_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={fowlkes_mallows_score(y, sbm_graph2_labels):.3f}\t")

    print(f"{Data} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={v_measure_score(y, dbscan.labels_):.3f}\t"
          f"MS={v_measure_score(y, ms.labels_):.3f}\t"
          f"Ag={v_measure_score(y, ward.labels_):.3f}\t"
          f"FCM={v_measure_score(y, labels):.3f}\t"
          f"SBM_array={v_measure_score(y, sbm_array_labels):.3f}\t"
          # f"SBM_graph={v_measure_score(y, sbm_graph_labels):.3f}\t"
          f"SBM_graph2={v_measure_score(y, sbm_graph2_labels):.3f}\t")


# try_metric(data_pca, labels, 4, 18, 22)
# compare_metrics_graph_vs_array_structure("Electrod1", data_pca, labels, 4, 18, 22)

# import plotly.express as px
#
# pca_ = PCA(n_components=3)
# data_pca = pca_.fit_transform(units_by_multitrodes[1][0])
# # print(np.array(data_pca).shape)
# fig1 = px.scatter_3d(x=data_pca[:, 0], y=data_pca[:, 1], z=data_pca[:, 2], color=np.array(labels[1]))
# fig1 = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=np.array(labels[1]))
# fig1.show()
#
# fig2 = px.scatter_3d(units_by_multitrodes[1][1], color=labels)
# fig3 = px.scatter_3d(units_by_multitrodes[1][2], color=labels)
# fig4 = px.scatter_3d(units_by_multitrodes[1][3], color=labels)
#
# fig2.show()
# fig3.show()
# fig4.show()
