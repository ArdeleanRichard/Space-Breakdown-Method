import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import struct

import functions.scatter_plot as sp

WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_CHANNELS = 33

def read_data_file(filename, data_type):
    """
    General reading method that will be called on more specific functions
    :param filename: name of the file
    :param data_type: usually int/float chosen by the file format (int/float - mentioned in spktwe)
    :return: data: data read from file
    """

    with open(filename, 'rb') as file:
        data = []
        read_val = file.read(4)
        data.append(struct.unpack(data_type, read_val)[0])

        while read_val:
            read_val = file.read(4)
            try:
                data.append(struct.unpack(data_type, read_val)[0])
            except struct.error:
                break

        return np.array(data)

def parse_ssd_file(dir_name):
    """
    Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
    represents the number of spikes in each channel
    @param dir_name: Path to directory that contains the files
    @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
    """
    for file_name in os.listdir(dir_name):
        full_file_name = dir_name + file_name
        if full_file_name.endswith(".ssd"):
            file = open(full_file_name, "r")
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = np.array(lines)
            index = np.where(lines == 'Number of spikes in each unit:')
            count = 1
            while str(lines[index[0][0]+count]).isdigit():
                count+=1
            spikes_per_unit = lines[index[0][0]+1:index[0][0]+count]

            unit_electrode = [i.strip('El_') for i in lines if str(i).startswith('El_')]
            unit_electrode = np.array(unit_electrode)

            return spikes_per_unit.astype('int'), unit_electrode.astype('int')


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


def read_timestamps(timestamp_filename):
    return read_data_file(timestamp_filename, 'i')


def read_waveforms(waveform_filename):
    return read_data_file(waveform_filename, 'f')


def read_event_timestamps(event_timestamps_filename):
    return read_data_file(event_timestamps_filename, 'i')


def read_event_codes(event_codes_filename):
    return read_data_file(event_codes_filename, 'i')


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
    data_on_unit = data_by_unit[unit - 1]
    data_on_unit = np.reshape(data_on_unit, (-1, length))

    return data_on_unit


def plot_spikes_on_unit(waveforms_by_unit, unit, show=False):
    waveforms_on_unit = get_data_from_unit(waveforms_by_unit, unit, WAVEFORM_LENGTH)
    plt.figure()
    plt.title(f"Spikes ({len(waveforms_on_unit)}) on unit {unit}")
    for i in range(0, len(waveforms_on_unit)):
        plt.plot(np.arange(len(waveforms_on_unit[i])), waveforms_on_unit[i])

    if show:
        plt.show()


def units_by_channel(unit_electrode, data, data_length):
    units_in_channels = []
    labels = []
    for i in range(NR_CHANNELS):
        units_in_channels.insert(0, [])
        labels.insert(0, [])

    for unit, channel in enumerate(unit_electrode):
        waveforms_on_unit = get_data_from_unit(data, unit+1, data_length)
        units_in_channels[channel-1].extend(waveforms_on_unit.tolist())
        labels[channel-1].extend(list(np.full((len(waveforms_on_unit), ), unit+1)))


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


def plot_sorted_data(title, data, labels, show=False):
    data = np.array(data)
    pca_2d = PCA(n_components=2)
    data_pca_2d = pca_2d.fit_transform(data)
    sp.plot(title, data_pca_2d, labels)
    if show==True:
        plt.show()


def plot_sorted_data_all_available_channels(units_in_channels, labels):
    for channel in range(NR_CHANNELS):
        if units_in_channels[channel] != [] and labels[channel] != labels:
            plot_sorted_data(f"Units in Channel {channel+1}", units_in_channels[channel], labels[channel])
    plt.show()