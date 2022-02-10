from sklearn.decomposition import PCA
import numpy as np

from functions.realdata_ssd_1electrode import parse_ssd_file, plot_sorted_data_all_available_channels, plot_spikes_on_unit
from functions.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from functions.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel, plot_sorted_data

DATASET_PATH = '../../data/kampff/c28/units/'

spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
WAVEFORM_LENGTH = 54
TIMESTAMP_LENGTH = 1
NR_CHANNELS = 1
unit_electrode = [1,1,1,1]

print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_electrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_electrode}")
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

event_timestamps = read_event_timestamps(event_timestamps_filename)
print(f"Event Timestamps found in file: {event_timestamps.shape}")
event_codes = read_event_codes(event_codes_filename)
print(f"Event Codes found in file: {event_codes.shape}")
# print(event_timestamps)
# print(event_codes)
print(f"Assert equality: {list(event_timestamps) == len(event_codes)}")
print("--------------------------------------------")

print(event_timestamps, len(event_timestamps))
print(event_codes, len(event_codes))
print(event_timestamps[event_codes == 1])
print(timestamps, len(timestamps))

print(waveforms_by_unit.shape)
print(waveforms_by_unit[0].shape)

# plot_spikes_on_unit(waveforms_by_unit, 0, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 1, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 2, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 3, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)

units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=NR_CHANNELS)

plot_sorted_data("", units_in_channels[0], labels[0], nr_dim=2, show=True)

intracellular_labels = np.zeros((len(timestamps)))
given_index = np.zeros((len(event_timestamps[event_codes == 1])))
for index, timestamp in enumerate(timestamps):
    for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
            given_index[index2] = 1
            intracellular_labels[index] = 1
            break

print(len(intracellular_labels))
print(np.count_nonzero(np.array(intracellular_labels)))

plot_sorted_data("", units_in_channels[0], intracellular_labels, nr_dim=2, show=True)




