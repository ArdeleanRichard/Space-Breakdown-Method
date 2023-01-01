import struct
import numpy as np


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


def read_timestamps(timestamp_filename):
    return read_data_file(timestamp_filename, 'i')


def read_waveforms(waveform_filename):
    return read_data_file(waveform_filename, 'f')


def read_event_timestamps(event_timestamps_filename):
    return read_data_file(event_timestamps_filename, 'i')


def read_event_codes(event_codes_filename):
    return read_data_file(event_codes_filename, 'i')


def read_data(channel_filename):
    return read_data_file(channel_filename, 'f')