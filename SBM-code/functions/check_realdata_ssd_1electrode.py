from sklearn.decomposition import PCA
import numpy as np

from functions.realdata_ssd_1electrode import parse_ssd_file, plot_sorted_data_all_available_channels, plot_spikes_on_unit
from functions.realdata_parsing import read_timestamps, read_waveforms
from functions.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel, plot_sorted_data

# DATASET_PATH = '../../data/M045_0005/'
# change NR_CHANNELS=33
# DATASET_PATH = '../../data/M045_SRCS_0009/'
DATASET_PATH = '../../data/M045_0009/'


spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_CHANNELS = 33

print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_electrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_electrode}")
print("--------------------------------------------")

print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")

timestamps = read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_unit)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_unit)}")

timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_unit))}")
print(f"Assert equality: {list(spikes_per_unit) == list(map(len, timestamps_by_unit))}")
print("--------------------------------------------")

waveforms = read_waveforms(waveform_file)
print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*{WAVEFORM_LENGTH}: {len(timestamps) * WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
waveform_lens = list(map(len, waveforms_by_unit))
print(f"Waveforms/{WAVEFORM_LENGTH} per channel should be equal: {[i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_unit) == [i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")


plot_spikes_on_unit(waveforms_by_unit, 1, show=True)
plot_spikes_on_unit(waveforms_by_unit, 3, show=True)


units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=NR_CHANNELS)
# for id, data in enumerate(units_in_channels):
#     print(id, len(data))
plot_sorted_data("", units_in_channels[0], labels[0], nr_dim=3, show=True)
plot_sorted_data("", units_in_channels[4], labels[4], nr_dim=3, show=True)
plot_sorted_data("", units_in_channels[5], labels[5], nr_dim=3, show=True)
plot_sorted_data("", units_in_channels[8], labels[8], nr_dim=3, show=True)
# plot_sorted_data_all_available_channels(units_in_channels, labels)



# TIMESTAMP FILTER FOR TETRODE TO HAVE THE SAME DATA
def timestamp_filter(tetrode_channels):
    t1 = np.array(timestamps_by_channel[tetrode_channels[0]]).flatten()
    t2 = np.array(timestamps_by_channel[tetrode_channels[1]]).flatten()
    t3 = np.array(timestamps_by_channel[tetrode_channels[2]]).flatten()
    t4 = np.array(timestamps_by_channel[tetrode_channels[3]]).flatten()
    print(len(t1))
    print(len(t2))
    print(len(t3))
    print(len(t4))

    delay = 500

    t1_index = []
    t2_index = []
    t3_index = []
    t4_index = []
    for ind in range(len(t1)):
        find_t2 = np.where((t2 <= t1[ind] + delay) & (t1[ind] - delay <= t2))
        find_t3 = np.where((t3 <= t1[ind] + delay) & (t1[ind] - delay <= t3))
        find_t4 = np.where((t4 <= t1[ind] + delay) & (t1[ind] - delay <= t4))
        if find_t2[0].size != 0 and find_t3[0].size != 0 and find_t4[0].size != 0:
            t1_index.append(ind)
            t2_index.append(find_t2[0][0])
            t3_index.append(find_t3[0][0])
            t4_index.append(find_t4[0][0])

    # print(t1[t1_index])
    # print(t2[t2_index])
    # print(t3[t3_index])
    # print(t4[t4_index])

    return np.array(t1_index), np.array(t2_index), np.array(t3_index), np.array(t4_index)

tetrode = [0, 4, 5, 8]

timestamps_by_channel, labels = units_by_channel(unit_electrode, timestamps_by_unit, data_length=TIMESTAMP_LENGTH, number_of_channels=NR_CHANNELS)
t1_index, t2_index, t3_index, t4_index = timestamp_filter(tetrode)

t1_units = np.array(units_in_channels[0])
t2_units = np.array(units_in_channels[4])
t3_units = np.array(units_in_channels[5])
t4_units = np.array(units_in_channels[8])

t1_labels = np.array(labels[0])
t2_labels = np.array(labels[4])
t3_labels = np.array(labels[5])
t4_labels = np.array(labels[8])

print(len(t1_labels[t1_index]))
print(len(t2_labels[t2_index]))
print(len(t3_labels[t3_index]))
print(len(t4_labels[t4_index]))

plot_sorted_data("", t1_units[t1_index], t1_labels[t1_index], nr_dim=3, show=True)
plot_sorted_data("", t2_units[t2_index], t2_labels[t2_index], nr_dim=3, show=True)
plot_sorted_data("", t3_units[t3_index], t3_labels[t3_index], nr_dim=3, show=True)
plot_sorted_data("", t4_units[t4_index], t4_labels[t4_index], nr_dim=3, show=True)
