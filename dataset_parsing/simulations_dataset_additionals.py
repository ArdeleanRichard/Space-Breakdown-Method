import numpy as np
from peakdetect import peakdetect
import matplotlib.pyplot as plt
from visualization import scatter_plot
from dataset_parsing.simulations_dataset import get_dataset_simulation, load_dictionaries


def get_dataset_simulation_features(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset with 2 chosen feaWWtures (amplitude and distance between min peaks)
    The amplitude can be used to find noise (noise has a lower amplitude than the other clusters, depending on similuation from 0.75->1)
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_features: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    spikes_features = np.empty((len(spikes), 2))

    for i in range(len(spikes)):
        # print(i)
        max_peaks, min_peaks = peakdetect(spikes[i], range(spike_length), lookahead=1)
        # print(max_peaks)
        # print(min_peaks)
        max_peaks = np.array(max_peaks)

        amplitude_information = np.argmax(max_peaks[:, 1])
        amplitude_position = max_peaks[amplitude_information][0]
        spike_amplitude = max_peaks[amplitude_information][1]

        spike_distance = 0

        if amplitude_position < min_peaks[0][0]:
            spike_distance = min_peaks[0][0] - 0
        else:
            for j in range(0, len(min_peaks)):
                if j + 1 >= len(min_peaks):
                    spike_distance = 79 - min_peaks[j][0]
                    # plt.figure()
                    # plt.plot(spikes[i])
                    # plt.savefig(f"./figures/FirstSpike{i}")
                    break
                else:
                    if min_peaks[j][0] < amplitude_position < min_peaks[j + 1][0]:
                        spike_distance = min_peaks[j + 1][0] - min_peaks[j][0]
                        break

        spikes_features[i] = [spike_amplitude, spike_distance]

        # if spike_amplitude < 0.5:
        #     plt.figure()
        #     plt.plot(spikes[i])
        #     plt.savefig(f"./figures/Noise{i},{spike_distance}")

    return spikes_features, labels



def get_dataset_simulation_noise(simNr, spike_length=79, noise_length=79):
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
    noise = noise_extract(data, start, spike_length, noise_length)

    return noise


def noise_extract(signal, spike_start, spike_length, noise_length):
    """
    Extract the noise from the signal knowing where the spikes start and their length
    :param signal: matrix - height for each point of the spikes
    :param spike_start: vector - each entry represents the first point of a spike
    :param spike_length: integer - constant, 79

    :returns spikes: matrix - each row contains 79 points of one spike
    """
    signal = np.array(signal)

    mask = np.ones((len(signal),))
    for i in range(len(spike_start)):
        mask[spike_start[i] - int(spike_length *1/2): spike_start[i] + int(spike_length *3/2)] = 0

    signal = signal[mask.astype(np.bool)]

    noise = np.reshape(signal[:-(len(signal) % spike_length)], (-1, spike_length))

    return noise



"""
----------------------------------------------------------
OBSOLETE
----------------------------------------------------------
"""


def getDatasetSim97Plots(spikes, spike_pca_2d, spikes_pca_3d, labels):
    # plot some spikes
    ind = np.random.randint(0, len(labels), [20])
    plt.plot(np.transpose(spikes[ind, :]))
    plt.show()

    # plot all spikes from one unit
    unit = 15
    ind = np.squeeze(np.argwhere(labels == unit))
    plt.plot(np.transpose(spikes[ind, :]))
    plt.title('Unit {}'.format(unit))
    plt.show()

    plotSimulation_PCA2D_grid(spike_pca_2d, labels)

    plotSimulation_PCA3D(spikes_pca_3d, labels)


def plotSimulation_PCA2D(spike_pca_2d, labels):
    # plot scatter of pca_nonoise
    plt.scatter(spike_pca_2d[:, 0], spike_pca_2d[:, 1], c=labels, marker='x', cmap='brg')
    plt.show()


def plotSimulation_PCA2D_grid(spike_pca_2d, labels):
    # plot scatter of pca_nonoise
    scatter_plot.plot_grid('Sim79Gridded', spike_pca_2d, labels + 1, 25, marker='x')
    plt.show()


def plotSimulation_PCA3D(spikes_pca_3d, labels):
    # plot scatter of pca_nonoise in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(spikes_pca_3d[:, 0], spikes_pca_3d[:, 1], spikes_pca_3d[:, 2], c=labels, marker='x', cmap='brg')
    plt.show()





