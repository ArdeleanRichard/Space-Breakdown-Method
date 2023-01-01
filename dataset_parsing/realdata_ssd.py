import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from visualization import scatter_plot as sp


def find_ssd_files(DATASET_PATH):
    """
    Searches in a folder for certain file formats and returns them
    :param DATASET_PATH: folder that contains files, looks for files that contain the data
    :return: returns the names of the files that contains data
    """
    timestamp_file = None
    waveform_file = None
    event_timestamps_filename = None
    event_codes_filename = None
    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith(".ssdst"):
            timestamp_file = DATASET_PATH + file_name
        if file_name.endswith(".ssduw"):
            waveform_file = DATASET_PATH + file_name
        if file_name.endswith(".ssdet"):
            event_timestamps_filename = DATASET_PATH + file_name
        if file_name.endswith(".ssdec"):
            event_codes_filename = DATASET_PATH + file_name

    return timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename


def separate_by_unit(spikes_per_unit, data, length):
    """
    Separates a data by spikes_per_unit, knowing that data are put one after another and unit after unit
    :param spikes_per_channel: list of lists - returned by parse_ssd_file
    :param data: timestamps / waveforms
    :param length: 1 for timestamps and 58 for waveforms
    :return:
    """
    separated_data = []
    sum=0
    for spikes_in_unit in spikes_per_unit:
        separated_data.append(data[sum*length: (sum+spikes_in_unit)*length])
        sum += spikes_in_unit
    return np.array(separated_data)


def get_data_from_unit(data_by_unit, unit, length):
    """
    Selects data by chosen unit
    :param data_by_channel: all the data of a type (all timestamps / all waveforms from all units)
    :param unit: receives inputs from 1 to NR_UNITS, stored in list with start index 0 (so its channel -1)
    :param length: 1 for timestamps and 58 for waveforms
    :return:
    """
    data_on_unit = data_by_unit[unit]
    data_on_unit = np.reshape(data_on_unit, (-1, length))

    return data_on_unit


def units_by_channel(unit_electrode, data, data_length, number_of_channels):
    units_in_channels = []
    labels = []
    for i in range(number_of_channels):
        units_in_channels.insert(0, [])
        labels.insert(0, [])

    for unit, channel in enumerate(unit_electrode):
        waveforms_on_unit = get_data_from_unit(data, unit, data_length)
        units_in_channels[channel-1].extend(waveforms_on_unit.tolist())
        labels[channel-1].extend(list(np.full((len(waveforms_on_unit), ), unit)))


    reset_labels = []
    for label_set in labels:
        if label_set != []:
            label_set = np.array(label_set)
            min_label = np.amin(label_set)
            label_set = label_set - min_label + 1
            reset_labels.append(label_set.tolist())
        else:
            reset_labels.append([])

    return units_in_channels, reset_labels


def plot_sorted_data(title, data, labels, nr_dim=2, show=False):
    data = np.array(data)
    pca_ = PCA(n_components=nr_dim)
    data_pca = pca_.fit_transform(data)
    sp.plot(title, data_pca, labels)
    if show==True:
        plt.show()
