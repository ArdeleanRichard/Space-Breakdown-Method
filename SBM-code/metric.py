import numpy as np


def remove_predicted_noise(true_labels, predicted_labels, noise_label):
    return true_labels[predicted_labels!=noise_label], predicted_labels[predicted_labels != noise_label]


def ss_metric(true_labels, predicted_labels, remove_noise=False):
    if isinstance(remove_noise, bool) and remove_noise == False:
        pass
    else:
        true_labels, predicted_labels = remove_predicted_noise(true_labels, predicted_labels, remove_noise)

    score = 0
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # array of bools of which in the predicted array are the current
        only_predicted_label = predicted_labels == predicted_label
        # array of true labels of the current predicted label
        true_labels_of_predicted = true_labels[only_predicted_label]

 
        # count of how many of the true label of current are in the array of true labels of the current predicted label
        # divide this by the number of predicted labels in the predicted array
        score += np.count_nonzero(true_labels_of_predicted == true_label)/np.count_nonzero(only_predicted_label)

    return score/len(true_labels)
