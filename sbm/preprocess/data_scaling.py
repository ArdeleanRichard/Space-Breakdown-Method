import numpy as np
from sklearn import preprocessing


def normalize_data_min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def choose_scale(spikes, scale_type):
    if scale_type == 'minmax':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    if scale_type == 'minmax_relu':
        spikes = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
    if scale_type == 'minmax_spp':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 4) - 3
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == '-1+1':
        spikes = spike_scaling_min_max(spikes, min_peak=-1, max_peak=1)
        # plot_spikes(spikes_scaled, title='scale_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'scaler':
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        # plot_spikes(spikes_scaled, title='scale_sklearn_0_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude':
        # SCALE IGNORE AMPLITUDE
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_no_amplitude', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude_add_amplitude':
        # SCALE IGNORE AMPLITUDE ADDED FEATURE AMPLITUDE
        amplitudes = np.amax(spikes, axis=1)
        amplitudes = amplitudes.reshape((-1,1))
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        spikes = np.hstack((spikes, amplitudes))
    elif scale_type == 'add_energy':
        # SCALED ADDED FEATURE ENERGY
        spikes_energy = get_spike_energy(spikes)
        spikes_energy = spikes_energy.reshape((-1,1))
        spikes = np.hstack((spikes, spikes_energy))
        # print(spikes.shape)
    elif scale_type == 'divide_amplitude':
        amplitudes = np.amax(spikes, axis=1)
        print(len(amplitudes))
        amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
        spikes = spikes / amplitudes
    elif scale_type == 'scale_no_energy_loss':
        # print(spikes)
        for spike in spikes:
            spike[spike < 0] = spike[spike < 0] / abs(np.amin(spike))
            spike[spike > 0] = spike[spike > 0] / np.amax(spike)

    return spikes


def min_max_scaling(X):
    """
    Min-Max scaling of the input, returning values within the [0, PN] interval
    :param X: matrix - the points of the dataset
    :param pn: int - the partioning number parameter

    :returns X_std: matrix - scaled dataset

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    """
    X_std = np.zeros_like(X)
    X_std[:, 0] = (X[:, 0] - np.amin(X[:, 0])) / (np.amax(X[:, 0]) - np.amin(X[:, 0]))
    X_std[:, 1] = (X[:, 1] - np.amin(X[:, 1])) / (np.amax(X[:, 1]) - np.amin(X[:, 1]))

    return X_std


def min_max_scaling_range(X, pn):
    X = min_max_scaling(X)
    X_std = X * pn

    return X_std


def spike_scaling_min_max(spikes, min_peak, max_peak):
    spikes_std = np.zeros_like(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scaling_ignore_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    for row in range(len(spikes)):
        min_peak = np.amin(spikes[row])
        max_peak = np.amax(spikes[row])
        spikes_std[row] = (spikes[row] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scale_largest_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    max_peak = np.amax(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = spikes[:, col] / max_peak

    return spikes_std


def get_spike_energy(spike):
    return np.sum(np.power(spike, 2), axis=1)