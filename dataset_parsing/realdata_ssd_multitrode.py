import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset_parsing.realdata_ssd import plot_sorted_data

NR_CHANNELS_MULTITRODE = 4
NR_MULTITRODES = 8

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

            unit_line = [i.strip('Multitrode ') for i in lines if str(i).startswith('Multitrode ')]
            unit_line_split = [str(ul).split(" ") for ul in unit_line]
            unit_line_split = np.array(unit_line_split)
            unit_multitrode = unit_line_split[:, 0]
            multitrode_channels = unit_line_split[:, 1]

            return spikes_per_unit.astype('int'), unit_multitrode.astype('int'), multitrode_channels


def split_multitrode(units_in_multitrode, multitrode_length, length):
    units_by_electrodes = []
    for units in units_in_multitrode:
        if len(units) != 0:
            units = np.array(units)
            units_by_electrode = []
            for step in range(0, multitrode_length, length):
                units_by_electrode.append(units[:, step:step+length])

            units_by_electrodes.append(units_by_electrode)
        else:
            units_by_electrodes.append([])

    return units_by_electrodes


def select_data(data, multitrode_nr, electrode_in_multitrode):
    return data[multitrode_nr][electrode_in_multitrode]


def plot_multitrode(data, labels, multitrode_nr, nr_electrodes, nr_dim=2):
    for nr in range(nr_electrodes):
        plot_sorted_data(f'Multitrode {multitrode_nr + 1} - Electrode {nr + 1}', data[multitrode_nr][nr], labels[multitrode_nr], nr_dim=nr_dim, show=True)


def plot_multitrode2(data, labels, multitrode_nr, nr_electrodes):
    for nr in range(nr_electrodes):
        plot_sorted_data(f'Multitrode {multitrode_nr + 1} - Electrode {nr + 1}', data[nr], labels, nr_dim=2, show=True)


def plot_multitrodes(data, labels, nr_multitrodes, nr_electrodes):
    for nr in range(nr_multitrodes):
        plot_multitrode2(data[nr], labels[nr], nr, nr_electrodes)