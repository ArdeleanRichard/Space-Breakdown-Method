import numpy as np

def apply_mask(data, mask):
    test = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j] != 0:
                test[i, j] = data[i, j]

    return test