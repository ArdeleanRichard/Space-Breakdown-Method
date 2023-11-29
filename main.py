import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, v_measure_score

import dataset_parsing.read_tins_m_data as ds
from validation.scores import purity_score
from validation.scs_metric import scs_metric
from visualization import scatter_plot as sp
import dataset_parsing.simulations_dataset as sds

import numpy as np
from sklearn.decomposition import PCA

from algorithms import SBM, ISBM
from dataset_parsing.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, plot_multitrode
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel
from visualization.scatter_plot_additionals import plot_spikes_by_clusters


def run_ISBM_graph_on_simulated_data():
    data, y = sds.get_dataset_simulation_pca_2d(4)
    pn = 10
    labels = ISBM.run(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=10) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

    pn = 25
    labels = ISBM.run(data, pn, ccThreshold=5, adaptivePN=True)
    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('ISBM(PN=25) on Sim4', data, pn, labels, marker='o', adaptivePN=True)

    plt.show()


def run_ISBM_graph_on_real_data():
    units_in_channel, labels = ds.get_tins_data()

    # for (i, pn) in list([(4, 25), (6, 40), (17, 15), (26, 30)]):
    for (i, pn) in list([(6, 40)]):
        print(i)
        data = units_in_channel[i-1]
        data = np.array(data)
        pca_2d = PCA(n_components=2)
        X = pca_2d.fit_transform(data)
        km_labels = labels[i-1]

        sp.plot('Ground truth', X, km_labels, marker='o')

        sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
        sp.plot_grid(f'ISBM on Channel {i}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)

        plot_spikes_by_clusters(data, sbm_graph_labels)

        km = KMeans(n_clusters=5).fit(X)
        sp.plot(f'K-means on Channel {i}', X, km.labels_, marker='o')

        plot_spikes_by_clusters(data,  km.labels_)

    plt.show()


def run_ISBM_graph_on_real_data_tetrode():
    DATASET_PATH = '../DATA/TINS/M017_Tetrode/ssd/'

    spikes_per_unit, unit_multitrode, _ = parse_ssd_file(DATASET_PATH)
    MULTITRODE_WAVEFORM_LENGTH = 232
    WAVEFORM_LENGTH = 58
    TIMESTAMP_LENGTH = 1
    NR_MULTITRODES = 8
    NR_ELECTRODES_PER_MULTITRODE = 4
    MULTITRODE_CHANNEL = 7

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)

    waveforms = read_waveforms(waveform_file)

    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, MULTITRODE_WAVEFORM_LENGTH)


    units_in_multitrode, labels = units_by_channel(unit_multitrode, waveforms_by_unit,
                                                   data_length=MULTITRODE_WAVEFORM_LENGTH,
                                                   number_of_channels=NR_MULTITRODES)
    units_by_multitrodes = split_multitrode(units_in_multitrode, MULTITRODE_WAVEFORM_LENGTH, WAVEFORM_LENGTH)


    # plot_multitrode(units_by_multitrodes, labels, MULTITRODE_CHANNEL, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
    labels = labels[MULTITRODE_CHANNEL]

    data_electrode1 = units_by_multitrodes[MULTITRODE_CHANNEL][0]

    data_electrode2 = units_by_multitrodes[MULTITRODE_CHANNEL][1]
    data_electrode3 = units_by_multitrodes[MULTITRODE_CHANNEL][2]
    data_electrode4 = units_by_multitrodes[MULTITRODE_CHANNEL][3]
    multitrode = np.hstack([data_electrode1, data_electrode2, data_electrode3, data_electrode4])

    pca_ = PCA(n_components=3)
    pca_vis = pca_.fit_transform(multitrode)

    pca_ = PCA(n_components=8)
    pca_data = pca_.fit_transform(multitrode)


    sbm_graph_labels = ISBM.run(pca_data, pn=20, ccThreshold=5, adaptivePN=True)
    sp.plot(f'ISBM on Tetrode (8 dimensions)', pca_vis, sbm_graph_labels, marker='o', alpha=0.5)
    plt.show()


def check_performances_against_scs(pn):
    SIM_NR = 4
    X, y = sds.get_dataset_simulation_pca_2d(SIM_NR)
    Title = f'Sim{SIM_NR}'
    nr_clusters = 5

    sp.plot(f'{Title} ground truth', X, y, marker='o', alpha=0.3)


    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(X)
    sp.plot(f'KMeans on {Title}', X, kmeans.labels_, marker='o', alpha=0.5)

    dbscan = DBSCAN(eps=0.1, min_samples=np.log(len(X))).fit(X)
    sp.plot(f'DBSCAN on {Title}', X, dbscan.labels_, marker='o', alpha=0.5)

    bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    sp.plot(f'MeanShift on {Title}', X, ms.labels_, marker='o', alpha=0.5)

    ward = AgglomerativeClustering(n_clusters=nr_clusters, linkage="ward").fit(X)
    sp.plot(f'AgglomerativeClustering on {Title}', X, ward.labels_, marker='o', alpha=0.5)

    my_model = FCM(n_clusters=nr_clusters)
    my_model.fit(X)
    labels = my_model.predict(X)
    sp.plot(f'FCM on {Title}', X, labels, marker='o', alpha=0.5)

    sbm_array_labels = SBM.best(X, pn, ccThreshold=5)

    # sp.plot_grid(f'SBM on {Title}', X, pn, sbm_array_labels, marker='o')
    sp.plot(f'SBM on {Title}', X, sbm_array_labels, marker='o', alpha=0.5)

    sbm_graph_labels = ISBM.run(X, pn, ccThreshold=5, adaptivePN=True)
    # sp.plot_grid(f'ISBM on {Title}', X, pn, sbm_graph_labels, marker='o', adaptivePN=True)
    sp.plot(f'ISBM on {Title}', X, sbm_graph_labels, marker='o', alpha=0.5)

    plt.show()


    print(f"{Title} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_rand_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_rand_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_rand_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_rand_score(y, labels):.3f}\t"
          f"SBM={adjusted_rand_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={adjusted_rand_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={adjusted_mutual_info_score(y, dbscan.labels_):.3f}\t"
          f"MS={adjusted_mutual_info_score(y, ms.labels_):.3f}\t"
          f"Ag={adjusted_mutual_info_score(y, ward.labels_):.3f}\t"
          f"FCM={adjusted_mutual_info_score(y, labels):.3f}\t"
          f"SBM={adjusted_mutual_info_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={adjusted_mutual_info_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={purity_score(y, dbscan.labels_):.3f}\t"
          f"MS={purity_score(y, ms.labels_):.3f}\t"
          f"Ag={purity_score(y, ward.labels_):.3f}\t"
          f"FCM={purity_score(y, labels):.3f}\t"
          f"SBM={purity_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={purity_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={fowlkes_mallows_score(y, dbscan.labels_):.3f}\t"
          f"MS={fowlkes_mallows_score(y, ms.labels_):.3f}\t"
          f"Ag={fowlkes_mallows_score(y, ward.labels_):.3f}\t"
          f"FCM={fowlkes_mallows_score(y, labels):.3f}\t"
          f"SBM={fowlkes_mallows_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={fowlkes_mallows_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={v_measure_score(y, dbscan.labels_):.3f}\t"
          f"MS={v_measure_score(y, ms.labels_):.3f}\t"
          f"Ag={v_measure_score(y, ward.labels_):.3f}\t"
          f"FCM={v_measure_score(y, labels):.3f}\t"
          f"SBM={v_measure_score(y, sbm_array_labels):.3f}\t"
          f"ISBM={v_measure_score(y, sbm_graph_labels):.3f}\t")

    print(f"{Title} - SCS: "
          f"KMeans={scs_metric(y, kmeans.labels_):.3f}\t"
          f"DBSCAN={scs_metric(y, dbscan.labels_):.3f}\t"
          f"MS={scs_metric(y, ms.labels_):.3f}\t"
          f"Ag={scs_metric(y, ward.labels_):.3f}\t"
          f"FCM={scs_metric(y, labels):.3f}\t"
          f"SBM={scs_metric(y, sbm_array_labels):.3f}\t"
          f"ISBM={scs_metric(y, sbm_graph_labels):.3f}\t")
    print()


def plot_data_result_mask(method_name, data, labelsMatrix, center_coords):
    fig= plt.figure(figsize=(24, 6))
    fig.suptitle(method_name, fontsize=16)
    ax = fig.add_subplot(1, 3, 1)

    ax.set_title("Initial Data")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Mask")
    im = ax.imshow(labelsMatrix, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    for center_coord in center_coords:
        plt.scatter(center_coord[1], center_coord[0], s=1, c='black', marker='o')
        # if cc['parent'] == -1:
        #     plt.scatter(coords[1], coords[0], s=1, c='white', marker='o')

    masked = apply_mask(data, labelsMatrix)

    # test = np.zeros_like(data)
    # for cc in cc_info:
    #     contour_points = cc['contour']
    #     for contour_point in contour_points:
    #             test[contour_point[0], contour_point[1]] = 1

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Segmentation")
    im = ax.imshow(masked, aspect='auto', cmap='jet', interpolation='none')
    plt.colorbar(im)
    # im = ax.imshow(test, aspect='auto', cmap='bone_r', interpolation='none', alpha=test)
    ax.invert_yaxis()


    plt.show()


def TFBM_and_plot(data):
    tfbm = TFBM(data.T, threshold="auto", merge=True, aspect_ratio=1, merge_factor=15)
    tfbm.fit(verbose=True, timer=True)

    center_coords = [(pi.center_coords[1], pi.center_coords[0]) for pi in tfbm.packet_infos]

    plot_data_result_mask("TFBM", data, tfbm.merged_labels_data.T, center_coords)


def load_atoms_synthetic_data():
    data_folder = "./DATA/toy/"
    file = "atoms-2.csv"

    f = open(data_folder+file, "r")
    intro = f.readlines()[:5]
    f.close()

    timeValues = []
    for str_time in intro[1].split(","):
        timeValues.append(float(str_time))

    frequencyValues = []
    for str_time in intro[3].split(","):
        frequencyValues.append(float(str_time))

    data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)

    spectrumData = Spectrum2D(timeValues=np.array(timeValues), frequencyValues=frequencyValues, powerValues=data)

    return data, spectrumData


if __name__ == '__main__':
    run_ISBM_graph_on_simulated_data()
    # run_ISBM_graph_on_real_data()
    # run_ISBM_graph_on_real_data_tetrode()
    # check_performances_against_scs(pn=10)
    # check_performances_against_scs(pn=25)

    # data, spectrumData = load_atoms_synthetic_data()
    # TFBM_and_plot(data)
