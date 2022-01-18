import numpy as np
from sklearn.decomposition import PCA

from functions.realdata_ssd_multitrode import parse_ssd_file, split_multitrode, select_data, plot_multitrode, plot_multitrodes
from functions.realdata_parsing import read_timestamps, read_waveforms, read_event_codes, read_event_timestamps
from functions.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel, plot_sorted_data

DATASET_PATH = '../../data/M046_0001_MT/'
# DATASET_PATH = '../../data/M045_RF_0008_19_MT/'
# DATASET_PATH = '../../data/M045_SRCS_0009_MT/'
# DATASET_PATH = '../../data/M045_DRCT_0015_MT/'

spikes_per_unit, unit_multitrode, _ = parse_ssd_file(DATASET_PATH)
MULTITRODE_WAVEFORM_LENGTH = 232
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
NR_MULTITRODES = 8
NR_ELECTRODES_PER_MULTITRODE = 4

print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_multitrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_multitrode}")
print("--------------------------------------------")

print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
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

event_timestamps = read_event_timestamps(event_timestamps_filename)
print(f"Event Timestamps found in file: {event_timestamps.shape}")
event_codes = read_event_codes(event_codes_filename)
print(f"Event Codes found in file: {event_codes.shape}")
# print(event_timestamps)
# print(event_codes)
print(f"Assert equality: {list(event_timestamps) == len(event_codes)}")
print("--------------------------------------------")

waveforms = read_waveforms(waveform_file)

# Check between stimulus
# print(event_timestamps)
# print(event_codes)
# print(waveforms.shape)
# timestamp_start = event_timestamps[1]
# timestamp_stop = event_timestamps[81]
# waveforms_reshaped = waveforms.reshape((-1,58*4))
# print(timestamps.shape)
# print(waveforms_reshaped.shape)
# print((timestamps > timestamp_start).shape)
# cond = np.logical_and(timestamps > timestamp_start, timestamps < timestamp_stop)
# waveforms = waveforms_reshaped[cond]
# waveforms = waveforms.reshape((1, -1))[0]
# print(waveforms.shape)

print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*{MULTITRODE_WAVEFORM_LENGTH}: {len(timestamps) * MULTITRODE_WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * MULTITRODE_WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, MULTITRODE_WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
waveform_lens = list(map(len, waveforms_by_unit))
print(f"Waveforms/{MULTITRODE_WAVEFORM_LENGTH} per channel should be equal: {[i//MULTITRODE_WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_unit) == [i//MULTITRODE_WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")

units_in_multitrode, labels = units_by_channel(unit_multitrode, waveforms_by_unit, data_length=MULTITRODE_WAVEFORM_LENGTH, number_of_channels=NR_MULTITRODES)
# for unit in units_in_multitrode:
#     print(len(unit))
units_by_multitrodes = split_multitrode(units_in_multitrode, MULTITRODE_WAVEFORM_LENGTH, WAVEFORM_LENGTH)
# for unit in units_by_multitrodes:
#     print(len(unit))

# data = select_data(data=units_by_multitrodes, multitrode_nr=0, electrode_in_multitrode=0)
# plot_multitrodes(units_by_multitrodes, labels, nr_multitrodes=NR_MULTITRODES, nr_electrodes=NR_ELECTRODES_PER_MULTITRODE)
plot_multitrode(units_by_multitrodes, labels, 1, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 3, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 5, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)
# plot_multitrode(units_by_multitrodes, labels, 7, NR_ELECTRODES_PER_MULTITRODE, nr_dim=3)

# import plotly.express as px
#
# pca_ = PCA(n_components=3)
# data_pca = pca_.fit_transform(units_by_multitrodes[1][0])
# # print(np.array(data_pca).shape)
# fig1 = px.scatter_3d(x=data_pca[:, 0], y=data_pca[:, 1], z=data_pca[:, 2], color=np.array(labels[1]))
# fig1 = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=np.array(labels[1]))
# fig1.show()
#
# fig2 = px.scatter_3d(units_by_multitrodes[1][1], color=labels)
# fig3 = px.scatter_3d(units_by_multitrodes[1][2], color=labels)
# fig4 = px.scatter_3d(units_by_multitrodes[1][3], color=labels)
#
# fig2.show()
# fig3.show()
# fig4.show()


