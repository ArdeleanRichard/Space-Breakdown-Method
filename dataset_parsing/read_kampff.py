import numpy as np

from dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, \
    read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel


def read_kampff_c37():
    DATASET_PATH = './datasets/kampff/c37/Spikes/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1]
    #*# unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

    intracellular_labels = np.zeros((len(timestamps)))
    # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
    # for index, timestamp in enumerate(timestamps):
    #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
    #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
    #             given_index[index2] = 1
    #             intracellular_labels[index] = 1
    #             break

    for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        indexes = []
        for index, timestamp in enumerate(timestamps):
            if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH:
                # given_index[index2] = 1
                indexes.append(index)

        if indexes != []:
            min = indexes[0]
            for i in range(1, len(indexes)):
                if timestamps[indexes[i]] < timestamps[min]:
                    min = indexes[i]
            intracellular_labels[min] = 1

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels


def read_kampff_c28():
    DATASET_PATH = './datasets/kampff/c28/units/'
    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

    intracellular_labels = np.zeros((len(timestamps)))
    # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
    # for index, timestamp in enumerate(timestamps):
    #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
    #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
    #             given_index[index2] = 1
    #             intracellular_labels[index] = 1
    #             break

    for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        indexes = []
        for index, timestamp in enumerate(timestamps):
            if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH:
                # given_index[index2] = 1
                indexes.append(index)

        if indexes != []:
            min = indexes[0]
            for i in range(1, len(indexes)):
                if timestamps[indexes[i]] < timestamps[min]:
                    min = indexes[i]
            intracellular_labels[min] = 1

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels