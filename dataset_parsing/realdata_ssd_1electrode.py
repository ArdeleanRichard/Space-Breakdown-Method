import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_parsing.realdata_ssd import get_data_from_unit, plot_sorted_data

TIMESTAMP_LENGTH = 1
NR_CHANNELS = 33


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


def plot_spikes_on_unit(waveforms_by_unit, unit, WAVEFORM_LENGTH, show=False):
    waveforms_on_unit = get_data_from_unit(waveforms_by_unit, unit, WAVEFORM_LENGTH)
    plt.figure()
    plt.title(f"Spikes ({len(waveforms_on_unit)}) on unit {unit}")
    for i in range(0, len(waveforms_on_unit)):
        plt.plot(np.arange(len(waveforms_on_unit[i])), waveforms_on_unit[i])

    if show:
        plt.show()



def plot_sorted_data_all_available_channels(units_in_channels, labels):
    for channel in range(NR_CHANNELS):
        if units_in_channels[channel] != [] and labels[channel] != labels:
            plot_sorted_data(f"Units in Channel {channel + 1}", units_in_channels[channel], labels[channel])
    plt.show()