import numpy as np
from matplotlib import pyplot as plt

from dataset_parsing.realdata_channel import parse_epd_file
from dataset_parsing.realdata_parsing import read_data

DATASET_PATH = '../../datasets/real_data/M014/classic/'
NR_CHANNELS = 33

file_name, channel_files = parse_epd_file(DATASET_PATH, 33)
print(file_name)
print(channel_files)

data_channels = []
for channel_file in channel_files[:5]:
    data_channels.append(read_data(DATASET_PATH + channel_file))

data_channels = np.array(data_channels)
print(data_channels.shape)

threshold = 5 * np.median(np.abs(data_channels[0]) / 0.6745)
print(threshold)

plt.title(f"Channel Data")
plt.plot(data_channels[0])
plt.show()