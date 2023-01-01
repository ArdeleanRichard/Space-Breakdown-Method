import numpy as np
from PyEMD import EMD
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import fft

from feature_extraction.wlt import discretewlt as dwt, wavelets as wlt
from feature_extraction import shape_features, derivatives as deriv
from feature_extraction.slt import superlets as slt
from dataset_parsing.simulations_dataset import get_dataset_simulation_pca_2d, get_dataset_simulation_pca_3d, \
    get_dataset_simulation
from feature_extraction import feature_extraction_methods as fe

def get_dataset_simulation_wavelets(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after wavelets on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = wlt.fd_wavelets(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_wavelets_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = wlt.fd_wavelets(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_3d = PCA(n_components=3)
    result_spikes = pca_3d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_superlets_2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = slt.slt(spikes, 2, 1.1)
    # result_spikes1 = slt.slt2(spikes, 5, 1.5)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_superlets_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = slt.slt(spikes, 2, 1.1)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_3d = PCA(n_components=3)
    result_spikes = pca_3d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_derivatives(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = deriv.compute_fdmethod(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    # result_spikes = result_spikes1

    return result_spikes, labels


def get_dataset_simulation_dwt2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after dwt on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = dwt.dwt_fd_method(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes




def get_dataset_simulation_derivatives_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 3 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_3d: matrix - the 3-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes = deriv.compute_fdmethod3d(spikes)

    return result_spikes, labels




def get_shape_phase_distribution_features(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                          spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    pca_2d = PCA(n_components=2)

    features = shape_features.get_shape_phase_distribution_features(spikes)
    features = pca_2d.fit_transform(features)
    print("Variance Ratio = ", np.sum(pca_2d.explained_variance_ratio_))

    return features, labels


def get_hilbert_features(sim_nr, spike_length=79, feature_reduction='derivativesPCA2D', align_to_peak=True,
                         normalize_spike=False, plot=False, spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)

    spikes_hilbert = hilbert(spikes)

    envelope = np.abs(spikes_hilbert)

    features = reduce_dimensionality(envelope, feature_reduction)
    return features, labels


def get_emd_signal_no_residuum_features(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    # Signal without residuum
    features = np.zeros((spikes.shape[0], spikes.shape[1]))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        features[i] = np.sum(IMFs, axis=0)

    features = reduce_dimensionality(features, method='derivativesPCA2D')
    return features, labels


def get_emd_imf_derivatives_features(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                     spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    features = np.zeros((spikes.shape[0], 8))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        f = np.array(deriv.compute_fdmethod(IMFs))

        if IMFs.shape[0] >= 4:
            features[i] = np.concatenate((f[0], f[1], f[2], f[3]))
        elif IMFs.shape[0] >= 3:
            features[i] = np.concatenate((f[0], f[1], f[2], [0, 0]))
        else:
            features[i] = np.concatenate((f[0], f[1], [0, 0], [0, 0]))

    features = reduce_dimensionality(features, method='PCA2D')
    return features, labels


def reduce_dimensionality(n_features, method='PCA2D'):
    if method == 'PCA2D':
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(n_features)
    elif method == 'PCA3D':
        pca_3d = PCA(n_components=3)
        features = pca_3d.fit_transform(n_features)
    elif method == 'derivatives':
        features = deriv.compute_fdmethod(n_features)
    elif method == 'derivativesPCA2D':
        features = deriv.compute_fdmethod(n_features)
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(features)
    else:
        features = []
    return features


def apply_feature_extraction_method(sim_nr, method_nr):
    if method_nr == 0:
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    elif method_nr == 1:
        X, y = get_dataset_simulation_pca_3d(sim_nr, align_to_peak=True)
    elif method_nr == 2:
        X, y = get_dataset_simulation_derivatives(sim_nr, align_to_peak=True)
    elif method_nr == 3:
        X, y = get_dataset_simulation_superlets_2d(sim_nr, align_to_peak=True)
    elif method_nr == 4:
        X, y = get_dataset_simulation_superlets_3d(sim_nr, align_to_peak=True)
    elif method_nr == 5:
        X, y = get_dataset_simulation_wavelets(sim_nr, align_to_peak=True)
    elif method_nr == 6:
        X, y = get_dataset_simulation_wavelets_3d(sim_nr, align_to_peak=True)
    elif method_nr == 7:
        X, y = get_dataset_simulation_dwt2d(sim_nr, align_to_peak=True)
    elif method_nr == 8:
        X, y = get_hilbert_features(sim_nr, align_to_peak=True)
    elif method_nr == 8:
        # X, y = EMD, TODO
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    else:
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    return X, y




def generate_dataset_from_simulations(simulations, simulation_labels, save=False):
    spikes = []
    labels = []
    index = 0
    for sim_index in np.arange(len(simulations)):
        s, l = get_dataset_simulation(simulations[sim_index], 79, True, False)
        for spike_index in np.arange(len(s)):
            for wanted_label in np.arange(len(simulation_labels[sim_index])):
                if simulation_labels[sim_index][wanted_label] == l[spike_index]:
                    spikes.append(s[spike_index])
                    labels.append(index + wanted_label)
        index = index + len(simulation_labels[sim_index])

    spikes = np.array(spikes)
    labels = np.array(labels)
    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    return spikes, labels

def get_dataset_simulation_emd_quartiles(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                         spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    features = np.zeros((spikes.shape[0], 12))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        hb = np.abs(hilbert(spike))
        ffts = fft.fft(IMFs)
        freq = fft.fftfreq(len(ffts[0]))
        t = np.arange(79)

        # Only a name
        IMFs = np.abs(ffts)
        # f = np.array(deriv.compute_fdmethod(IMFs))

        f1 = np.array([np.percentile(IMFs[0], 25), np.percentile(IMFs[0], 50), np.percentile(IMFs[0], 75)])
        f2 = np.array([np.percentile(IMFs[1], 25), np.percentile(IMFs[1], 50), np.percentile(IMFs[1], 75)])
        f3 = np.array([0, 0, 0])
        f4 = np.array([0, 0, 0])
        if IMFs.shape[0] >= 3:
            f3 = np.array([np.percentile(IMFs[2], 25), np.percentile(IMFs[2], 50), np.percentile(IMFs[2], 75)])
        if IMFs.shape[0] >= 4:
            f4 = np.array([np.percentile(IMFs[3], 25), np.percentile(IMFs[3], 50), np.percentile(IMFs[3], 75)])

        # print(np.concatenate((np.array([f1, f2, f3, f4]))))
        features[i] = np.concatenate((np.array([f1, f2, f3, f4])))
        # print(freq)
        # plt.plot(freq, fft1.real, freq, fft1.imag)
        # plt.show()
        # plt.clf()
        # plt.plot(freq, fft2.real, freq, fft2.imag)
        # plt.show()

    features = fe.reduce_dimensionality(features, method='PCA2D')
    return features, labels
