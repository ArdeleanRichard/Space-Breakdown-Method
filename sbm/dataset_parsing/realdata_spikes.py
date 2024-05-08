import os

import numpy as np
import matplotlib.pyplot as plt

TIMESTAMP_LENGTH = 1


def parse_spktwe_file(dir_name, NR_CHANNELS = 32):
    """
    Function that parses the .spktwe file from the directory and returns an array of length=nr of channels and each value
    represents the number of spikes in each channel
    @param dir_name: Path to directory that contains the files
    @return: spikes_per_channel: an array of length=nr of channels and each value is the number of spikes on that channel
    """
    for file_name in os.listdir(dir_name):
        full_file_name = dir_name + file_name
        if full_file_name.endswith(".spktwe"):
            file = open(full_file_name, "r")
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = np.array(lines)
            index_line = np.where(lines == 'Number of stored data channels:')
            nr_chn = lines[index_line[0][0]+1].astype(np.int)
            print(nr_chn)

            index_line = np.where(lines == 'List with the names of stored channels:')
            stored_chns = lines[index_line[0][0]+1:index_line[0][0]+1 + nr_chn]
            spikes_per_all_channel = np.zeros((NR_CHANNELS,))
            # print(spikes_per_all_channel)

            index = np.where(lines == 'Number of spikes in each stored channel::')
            spikes_per_channels = lines[index[0][0]+1:index[0][0]+1+nr_chn]

            i = 0
            for stored_chn, spikes_per_channel in zip(stored_chns, spikes_per_channels):
                if stored_chn.replace("El_", "").isnumeric():
                    stored_chn = int(stored_chn.replace("El_", "")) - 1
                else:
                    stored_chn = i
                spikes_per_all_channel[stored_chn] = int(spikes_per_channel)
                i+=1
            # print(spikes_per_all_channel.astype('int'))
            # index = (array([44], dtype=int64),)

            return spikes_per_all_channel.astype('int')


def find_waverform_files(DATASET_PATH):
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
        if file_name.endswith(".spiket"):
            timestamp_file = DATASET_PATH + file_name
        if file_name.endswith(".spikew"):
            waveform_file = DATASET_PATH + file_name
        if file_name.endswith(".eventt"):
            event_timestamps_filename = DATASET_PATH + file_name
        if file_name.endswith(".eventc"):
            event_codes_filename = DATASET_PATH + file_name

    return timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename


def separate_by_channel(spikes_per_channel, data, length):
    """
    Separates a data by spikes_per_channel, knowing that data are put one after another and channel after channel
    :param spikes_per_channel: list of lists - returned by parse_spktwe_file
    :param data: timestamps / waveforms
    :param length: 1 for timestamps and 58 for waveforms
    :return:
    """
    separated_data = []
    sum=0
    for spikes_in_channel in spikes_per_channel:
        separated_data.append(data[sum*length: (sum+spikes_in_channel)*length])
        sum += spikes_in_channel
    return np.array(separated_data)


def get_data_from_channel(data_by_channel, channel, length):
    """
    Selects data by chosen channel
    :param data_by_channel: all the data of a type (all timestamps / all waveforms from all channels)
    :param channel: receives inputs from 1 to NR_CHANNELS, stored in list with start index 0 (so its channel -1)
    :param length: 1 for timestamps and 58 for waveforms
    :return:
    """
    data_on_channel = data_by_channel[channel - 1]
    data_on_channel = np.reshape(data_on_channel, (-1, length))

    return data_on_channel


def plot_spikes_on_channel(waveforms_by_channel, channel, WAVEFORM_LENGTH, show=False):
    waveforms_on_channel = get_data_from_channel(waveforms_by_channel, channel, WAVEFORM_LENGTH)
    plt.figure()
    plt.title(f"Spikes ({len(waveforms_on_channel)}) on channel {channel}")
    for i in range(0, len(waveforms_on_channel)):
        plt.plot(np.arange(len(waveforms_on_channel[i])), waveforms_on_channel[i])

    if show:
        plt.show()


def plot_all_spikes_by_channel(waveforms_by_channel, WAVEFORM_LENGTH):
    for index in range(len(waveforms_by_channel)):
        plot_spikes_on_channel(waveforms_by_channel, index+1, WAVEFORM_LENGTH, show=False)
    plt.show()

#
# def sum_until_channel(channel):
#     ch_sum = 0
#     for i in units_per_channel[:channel]:
#         ch_sum += np.sum(np.array(i)).astype(int)
#
#     return ch_sum
#
#
# def get_spike_units(waveform, channel, plot_spikes=False):
#     spikes = []  # np.zeros((np.sum(units_per_channel[channel]), 58))
#
#     #  Waveform is a continuous array, every 58 points is a new spike start index
#     for i in range(0, len(waveform), 58):
#         spikes.append(waveform[i: i + 58])  # Add spikes to a regular array, easier to work with
#
#     left_limit_spikes = sum_until_channel(channel)
#     right_limit_spikes = left_limit_spikes + np.sum(np.array(units_per_channel[channel]))
#     print(left_limit_spikes, right_limit_spikes)
#     spikes = spikes[left_limit_spikes: right_limit_spikes]
#
#     if plot_spikes:
#         for i in range(0, len(spikes), 1000):
#             plt.plot(np.arange(58), -spikes[i])
#         plt.show()
#
#     # Put labels into a similar array with spikes
#     labels = np.array([])
#     for i, units in enumerate(units_per_channel[channel]):
#         labels = np.append(labels, np.repeat(i, units))
#
#     return spikes, labels.astype(int)
#
#
# def get_spike_units(waveform, channel, plot_spikes=False):
#     spikes = []
#     new_spikes = np.zeros((np.sum(units_per_channel[channel]), 58))
#
#     for i in range(0, len(waveform), 58):
#         spikes.append(waveform[i: i + 58])
#
#     left_limit_spikes = sum_until_channel(channel)
#     right_limit_spikes = left_limit_spikes + np.sum(np.array(units_per_channel[channel]))
#     # print(left_limit_spikes, right_limit_spikes)
#     spikes = spikes[left_limit_spikes: right_limit_spikes]
#
#     if plot_spikes:
#         for i in range(0, len(spikes), 1000):
#             plt.plot(np.arange(58), -spikes[i])
#         plt.show()
#
#     labels = np.array([])
#     for i, units in enumerate(units_per_channel[channel]):
#         labels = np.append(labels, np.repeat(i, units))
#
#     for i, units in enumerate(units_per_channel[channel]):
#         left_lim = sum_until_channel(channel)
#         right_lim = left_lim + units
#
#         spike_index = 0
#         for j in range(len(spikes)):
#             # print(j, left_lim, right_lim)
#             new_spikes[spike_index] = spikes[j]
#             spike_index += 1
#         # scaler = StandardScaler()
#         # spikes = scaler.fit_transform(spikes)
#     return new_spikes, labels.astype(int)
#
#
# def plot_spikes_per_unit(waveforms):
#     for channel in range(0, len(units_per_channel)):
#         if len(units_per_channel[channel]) > 0:
#             spikes, labels = get_spike_units(waveforms, channel)
#             start = 0
#             unit_pos = 0
#             for unit in units_per_channel[channel]:
#                 print(unit)
#                 print(start + unit)
#                 scatter_plot.plot_spikes(-spikes[start:start + unit],
#                                          title=f"Channel{channel}Cluster{unit_pos}")
#                 start += unit
#                 unit_pos += 1
#
#
#
# def real_dataset(waveform_filename, channel, feature_extraction_method, dim_reduction_method):
#     waveforms = read_waveforms(waveform_filename)
#     spikes, labels = get_spike_units(waveforms, channel=channel)
#
#     spikes2d = fe.apply_feature_extraction_method(spikes, feature_extraction_method, dim_reduction_method)
#     scatter_plot.plot_clusters(spikes2d, labels,
#                                f'channel{channel}_{feature_extraction_method}_{dim_reduction_method}',
#                                save_folder='real_data')
#     plt.show()
#
#
# def extract_spikes(timestamps, waveform, channel):
#     left_limit = np.sum(spikes_per_channel[:channel])
#     right_limit = left_limit + spikes_per_channel[channel]
#     timestamps = timestamps[left_limit:right_limit]
#     # waveform = waveform[channel * 36297600: (channel + 1) * 36297600]
#
#     print(waveform.shape)
#
#     spikes = np.zeros((spikes_per_channel[channel], 58))
#     print(spikes.shape)
#     for index in range(len(timestamps)):
#         # print('index', index)
#         # print(timestamps[index])
#         # print(timestamps[index] + 58)
#         # print(waveform.shape)
#         # print()
#         spikes[index] = waveform[timestamps[index]: timestamps[index] + 58]
#     print(spikes.shape)
#     # print(spikes[-2].shape)
#
#     peak_ind = np.argmin(spikes, axis=1)
#     # avg_peak = np.floor(np.mean(peak_ind))
#     timestamps = timestamps - (19 - peak_ind)
#     timestamps = timestamps.astype(int)
#
#     spikes = []
#     for i in range(len(timestamps)):
#         spikes.append(waveform[timestamps[i]:timestamps[i] + 58])
#         plt.plot(np.arange(58), -spikes[i])
#     plt.show()
#
# def extract_spikes2(timestamps, waveform, channel):
#     left_limit = np.sum(spikes_per_channel[:channel])
#     right_limit = left_limit + spikes_per_channel[channel]
#     waveform_channel = waveform[left_limit:right_limit]
#     print(left_limit)
#     print(right_limit)
#
#     spikes = []
#     spike_index = 0
#     for i in range(int(left_limit), int(right_limit), 58):
#         print(len(waveform[i:i + 58]))
#         spikes.append(waveform_channel[i:i + 58])
#         print(spike_index)
#         spike_index += 1
#     # plt.show()
#
#     spikes = []
#     for i in range(0, spikes_per_channel[channel]):
#         spikes.append(waveform_channel[i * 58:(i + 1) * 58])
#
#     for i in range(0, 10):
#         plt.plot(np.arange(58), -spikes[i])
#     plt.show()
#
#
# def extract_spikes3(waveform, channel):
#     left_limit = np.sum(spikes_per_channel[:channel])
#     right_limit = left_limit + spikes_per_channel[channel]
#
#     spikes = []
#     for i in range(0, len(waveform), 58):
#         spikes.append(waveform[i:i + 58])
#     print(len(spikes))
#     for i in range(0, len(spikes), 1000):
#         plt.plot(np.arange(58), -spikes[i])
#     plt.show()
#
#     return spikes[left_limit:right_limit]
#
#
# def extract_spikes4(waveform):
#     spikes = []
#     for i in range(0, len(waveform), 58):
#         spikes.append(waveform[i:i + 58])
#     print(len(spikes))
#     for i in range(0, len(spikes), 1000):
#         plt.plot(np.arange(58), -spikes[i])
#     plt.show()
#
#     pca2d = PCA(n_components=2)
#     # X = pca2d.fit_transform(spikes[0:int(np.floor(spikes_per_channel[2]/58))])
#     X = pca2d.fit_transform(spikes[0:11837])
#     plt.scatter(X[:, 0], X[:, 1], marker='o', edgecolors='k')
#     plt.show()
#
#     pca3d = PCA(n_components=3)
#     X = pca3d.fit_transform(spikes[0:11837])
#     fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2])
#     fig.update_layout(title="Ground truth for channel 1")
#     fig.show()
#
#     der_spikes = derivatives.compute_fdmethod(spikes[0:11837])
#     plt.scatter(der_spikes[:, 0], der_spikes[:, 1], marker='o', edgecolors='k')
#     plt.show()
#
#
# def fe_per_channels():
#     # waveforms_ = read_waveforms('./datasets/real_data/M017_004_5stdv/Units/M017_0004_5stdv.ssduw')
#     # channel_list = [1, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 25, 27, 28, 32]
#     waveforms_ = read_waveforms('./datasets/real_data/M017_004_3stdv/Units/M017_S001_SRCS3L_25,50,100_0004.ssduw')
#     channel_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26, 28, 29, 30, 31, 32]
#     for ch in range(5, 6):
#         spikes, labels = get_spike_units(waveforms_, channel=channel_list[ch])
#         spikes2d = fe.apply_feature_extraction_method(spikes, 'pca3d')
#         sp.plot_clusters(spikes2d, labels, 'pca3d channel_%d' % channel_list[ch], 'real_data')
#         plt.show()