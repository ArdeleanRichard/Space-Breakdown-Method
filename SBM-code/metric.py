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


def ss_metric_unweighted(true_labels, predicted_labels, remove_noise=False):
    if isinstance(remove_noise, bool) and remove_noise == False:
        pass
    else:
        true_labels, predicted_labels = remove_predicted_noise(true_labels, predicted_labels, remove_noise)

    score = 0
    for unique_predicted_label in np.unique(predicted_labels):
        only_predicted_label = predicted_labels == unique_predicted_label
        true_labels_of_predicted = true_labels[only_predicted_label]

        test = np.unique(true_labels_of_predicted, return_counts=True)
        # print(unique_predicted_label,
        #       np.count_nonzero(true_labels_of_predicted == test[0][np.argmax(test[1])]) / np.count_nonzero(only_predicted_label),
        #       test[0][np.argmax(test[1])],
        #       test[0], np.argmax(test[1]))

        #print(np.count_nonzero(true_labels_of_predicted == test[0][np.argmax(test[1])]))
        #print(np.amax(test[1]))

        print(f"{np.amax(test[1]):.3f} / {np.count_nonzero(only_predicted_label):.3f} = {np.amax(test[1])/np.count_nonzero(only_predicted_label):.3f}")
        score += np.amax(test[1]) / np.count_nonzero(only_predicted_label)

    return score / len(np.unique(predicted_labels))


def ss_metric_unweighted2(true_labels, predicted_labels, remove_noise=False):
    if isinstance(remove_noise, bool) and remove_noise == False:
        pass
    else:
        true_labels, predicted_labels = remove_predicted_noise(true_labels, predicted_labels, remove_noise)

    score = 0
    for unique_true_label in np.unique(true_labels):
        only_true_label = true_labels == unique_true_label
        predicted_labels_of_true = predicted_labels[only_true_label]

        test = np.unique(predicted_labels_of_true, return_counts=True)
        # print(unique_true_label,
        #       np.count_nonzero(predicted_labels_of_true == test[0][np.argmax(test[1])]) / np.count_nonzero(only_true_label),
        #       test[0][np.argmax(test[1])],
        #       test[0], np.argmax(test[1]))

        score += np.count_nonzero(predicted_labels_of_true == test[0][np.argmax(test[1])]) / np.count_nonzero(only_true_label)

    return score / len(np.unique(true_labels))
