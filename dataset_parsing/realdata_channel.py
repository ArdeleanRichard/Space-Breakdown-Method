import os

import numpy as np


def parse_epd_file(dir_name, NR_CHANNELS = 32):
    """
    Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
    represents the number of spikes in each channel
    @param dir_name: Path to directory that contains the files
    @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
    """
    for file_name in os.listdir(dir_name):
        full_file_name = dir_name + file_name
        if full_file_name.endswith(".epd"):
            file = open(full_file_name, "r")
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = np.array(lines)
            index = np.where(lines == 'List with filenames that hold individual channel samples (32 bit IEEE 754-1985, single precision floating point; amplitudes are measured in uV):')
            # index = (array([44], dtype=int64),)

            channel_files = lines[index[0][0]+1:index[0][0]+1+NR_CHANNELS]

            return file_name, channel_files

