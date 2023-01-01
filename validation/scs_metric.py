import numpy as np

def scs_metric(true_labels, predicted_labels):
    score = 0
    for unique_true_label in np.unique(true_labels):
        only_true_label = true_labels == unique_true_label
        predicted_labels_of_true = predicted_labels[only_true_label]

        predicted_unique_labels = np.unique(predicted_labels_of_true, return_counts=True)[0]
        predicted_counts = np.unique(predicted_labels_of_true, return_counts=True)[1]

        predicted_label_of_true = predicted_unique_labels[np.argmax(predicted_counts)]

        score += np.amax(predicted_counts) / np.count_nonzero(predicted_labels == predicted_label_of_true)
    return score / len(np.unique(true_labels))
