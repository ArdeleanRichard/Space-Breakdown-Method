import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from ..constants import DATA_FOLDER_PATH, SIM_DATA_FOLDER_PATH


def load_dictionaries(simNr):
    simulation_dictionary = loadmat(SIM_DATA_FOLDER_PATH + '/simulation_' + str(simNr) + '.mat')
    ground_truth_dictionary = loadmat(SIM_DATA_FOLDER_PATH + '/ground_truth.mat')
    return simulation_dictionary, ground_truth_dictionary


def get_dataset_simulation_pca_2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_2d: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    # apply pca_nonoise
    pca_2d = PCA(n_components=2)
    spikes_pca_2d = pca_2d.fit_transform(spikes)
    # getDatasetSimulationPlots(spikes, spikes_pca_2d, spikes_pca_3d, labels)

    # np.save('79_ground_truth', label)
    # np.save('79_x', spikes_reduced[:, 0])
    # np.save('79_y', spikes_reduced[:, 1])

    return spikes_pca_2d, labels


# spike extraction options
# original sampling rate 96KHz, with each waveform at 316 points(dimensions/features)
# downsampled to 24KHz, (regula-3-simpla) => 79 points (de aici vine 79 de mai jos)
def get_dataset_simulation_pca_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
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
    # apply pca_nonoise
    pca_3d = PCA(n_components=3)
    spikes_pca_3d = pca_3d.fit_transform(spikes)

    return spikes_pca_3d, labels


def get_dataset_simulation(simNr, spike_length=79, align_to_peak=True, normalize_spike=False, scale_spike=False):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary, ground_truth_dictionary = load_dictionaries(simNr)

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    # each spike will contain the first 79 points from the data after it has started
    spikes = spike_preprocess(data, start, spike_length, align_to_peak, normalize_spike, scale_spike)

    return spikes, labels


def get_signal_simulation(simNr):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary, ground_truth_dictionary = load_dictionaries(simNr)

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    return data, start, labels


def spike_preprocess(signal, spike_start, spike_length, align_to_peak, normalize_spikes, scale_spikes, index_to_align_peak=39):
    spikes = spike_extract(signal, spike_start, spike_length)

    if scale_spikes == True:
        spikes_std = np.zeros_like(spikes)
        min_peak = np.amin(spikes)
        max_peak = np.amax(spikes)
        for col in range(len(spikes[0])):
            spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)
        spikes = spikes_std

    # align to max
    if isinstance(align_to_peak, bool) and align_to_peak:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmax(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        avg_peak = np.floor(np.mean(peak_ind))
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (avg_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)
    # align_to_peak == 2 means that it will align the maximum point (Na+ polarization) to index 39 (the middle)
    elif align_to_peak == 2:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmax(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        # index_to_align_peak = 39
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (index_to_align_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)
    # align_to_peak == 3 means that it will align the minimum point (K+ polarization) to index 39 (the middle)
    elif align_to_peak == 3:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmin(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        # index_to_align_peak = 39
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (index_to_align_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)

    # normalize spikes using Z-score: (value - mean)/ standard deviation
    if normalize_spikes:
        normalized_spikes = [(spike - np.mean(spike)) / np.std(spike) for spike in spikes]
        normalized_spikes = np.array(normalized_spikes)
        return normalized_spikes
    return spikes


def spike_extract(signal, spike_start, spike_length):
    """
    Extract the spikes from the signal knowing where the spikes start and their length
    :param signal: matrix - height for each point of the spikes
    :param spike_start: vector - each entry represents the first point of a spike
    :param spike_length: integer - constant, 79

    :returns spikes: matrix - each row contains 79 points of one spike
    """
    spikes = np.zeros([len(spike_start), spike_length])

    for i in range(len(spike_start)):
        spikes[i, :] = signal[spike_start[i]: spike_start[i] + spike_length]

    return spikes




