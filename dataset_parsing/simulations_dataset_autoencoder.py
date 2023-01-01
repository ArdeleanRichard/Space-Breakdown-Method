from dataset_parsing.simulations_dataset import get_dataset_simulation
import numpy as np

from preprocess.data_fft import apply_fft_on_data, apply_fft_windowed_on_data


def stack_simulations_array(number_array, randomize=False, no_noise=False):
    spikes = np.empty((79,))
    labels = np.empty((1,))
    for i in range(0, len(number_array)):
        if i == 25 or i == 44:
            continue
        new_spikes, new_labels = get_dataset_simulation(simNr=number_array[i])

        #remove noise label == 0
        if no_noise == True:
            new_spikes = new_spikes[new_labels != 0]
            new_labels = new_labels[new_labels != 0]

        if randomize == True:
            subsample = create_subsamples(new_spikes, new_labels,  percentage=80, nr_of_subsamples=1)
            new_spikes, new_labels = subsample[0]

        new_spikes = np.array(new_spikes)
        new_labels = np.array(new_labels)
        spikes = np.vstack((spikes, new_spikes))
        labels = np.hstack((labels, new_labels))

    # delete np.empty initialization
    return spikes[1:], labels[1:]


def split_data(spikes_list, labels_list, new_spikes, new_labels, count):
    cluster_train_spikes = np.empty((79,))
    cluster_train_labels = np.empty((1,))
    cluster_test_spikes = np.empty((79,))
    cluster_test_labels = np.empty((1,))

    unique = np.unique(new_labels)
    for cluster_index in unique:
        cluster_spikes = new_spikes[new_labels == cluster_index]
        random_train = np.random.choice(range(len(cluster_spikes)), len(cluster_spikes) // 2, replace=False)

        all_list = np.arange(0, len(cluster_spikes))
        random_test = np.setdiff1d(all_list, random_train)

        new_train_spikes = new_spikes[random_train, :]
        new_train_labels = new_labels[random_train]
        new_test_spikes = new_spikes[random_test, :]
        new_test_labels = new_labels[random_test]

        cluster_train_spikes = np.vstack((cluster_train_spikes, new_train_spikes))
        cluster_train_labels = np.hstack((cluster_train_labels, new_train_labels))
        cluster_test_spikes = np.vstack((cluster_test_spikes, new_test_spikes))
        cluster_test_labels = np.hstack((cluster_test_labels, new_test_labels))

    cluster_train_spikes = cluster_train_spikes[1:]
    cluster_train_labels = cluster_train_labels[1:]
    cluster_test_spikes = cluster_test_spikes[1:]
    cluster_test_labels = cluster_test_labels[1:]

    if count == 1:
        spikes_list.append(cluster_train_spikes)
        spikes_list.append(cluster_test_spikes)
        labels_list.append(cluster_train_labels)
        labels_list.append(cluster_test_labels)
    else:
        split_data(spikes_list, labels_list, cluster_train_spikes, cluster_train_labels, count - 1)
        split_data(spikes_list, labels_list, cluster_test_spikes, cluster_test_labels, count - 1)


def split_stack_simulations(min, max, no_noise=False, alignment=True, normalize=False, scale=False):
    count = 4

    spikes_list = []
    labels_list = []
    for i in range(2**count):
        spikes_list.append(np.empty((79,)))
        labels_list.append(np.empty((1,)))

    print(len(spikes_list))

    for simulation_number in range(min, max):
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44:
            continue
        new_spikes, new_labels = get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment,
                                                        normalize_spike=normalize, scale_spike=scale)

        sim_spikes_list = []
        sim_labels_list = []

        # remove noise label == 0
        if no_noise == True:
            new_spikes = new_spikes[new_labels != 0]
            new_labels = new_labels[new_labels != 0]

        split_data(sim_spikes_list, sim_labels_list, new_spikes, new_labels, count)
        for i in range(len(list(zip(sim_spikes_list, sim_labels_list)))):
            spikes_list[i] = np.vstack((spikes_list[i], sim_spikes_list[i]))
            labels_list[i] = np.hstack((labels_list[i], sim_labels_list[i]))

    for i in range(len(list(zip(spikes_list, labels_list)))):
        spikes_list[i] = spikes_list[i][1:]
        labels_list[i] = labels_list[i][1:]


    return spikes_list, labels_list


def stack_simulations_split_train_test(min, max, no_noise=False, alignment=True, normalize=False, scale=False):
    train_spikes = np.empty((79,))
    train_labels = np.empty((1,))
    test_spikes = []
    test_labels = []
    for simulation_number in range(min, max):
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44:
            continue
        new_spikes, new_labels = get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment, normalize_spike=normalize, scale_spike=scale)

        #remove noise label == 0
        if no_noise == True:
            new_spikes = new_spikes[new_labels != 0]
            new_labels = new_labels[new_labels != 0]

        cluster_train_spikes = np.empty((79,))
        cluster_train_labels = np.empty((1,))
        cluster_test_spikes = np.empty((79,))
        cluster_test_labels = np.empty((1,))

        unique = np.unique(new_labels)
        for cluster_index in unique:
            cluster_spikes = new_spikes[new_labels == cluster_index]
            random_train = np.random.choice(range(len(cluster_spikes)), len(cluster_spikes) // 2, replace=False)

            all_list = np.arange(0, len(cluster_spikes))
            random_test = np.setdiff1d(all_list, random_train)

            new_train_spikes = new_spikes[random_train, :]
            new_train_labels = new_labels[random_train]
            new_test_spikes = new_spikes[random_test, :]
            new_test_labels = new_labels[random_test]

            cluster_train_spikes = np.vstack((cluster_train_spikes, new_train_spikes))
            cluster_train_labels = np.hstack((cluster_train_labels, new_train_labels))
            cluster_test_spikes = np.vstack((cluster_test_spikes, new_test_spikes))
            cluster_test_labels = np.hstack((cluster_test_labels, new_test_labels))

        cluster_train_spikes = cluster_train_spikes[1:]
        cluster_train_labels = cluster_train_labels[1:]
        cluster_test_spikes = cluster_test_spikes[1:]
        cluster_test_labels = cluster_test_labels[1:]

        train_spikes = np.vstack((train_spikes, cluster_train_spikes))
        train_labels = np.hstack((train_labels, cluster_train_labels))
        test_spikes.append(cluster_test_spikes)
        test_labels.append(cluster_test_labels)

    # delete np.empty initialization
    return train_spikes[1:], train_labels[1:], np.array(test_spikes), np.array(test_labels)


def stack_simulations_range(min, max, randomize=False, no_noise=False, alignment=True):
    spikes = np.empty((79,))
    labels = np.empty((1,))
    for simulation_number in range(min, max):
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
            continue
        new_spikes, new_labels = get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

        #remove noise label == 0
        if no_noise == True:
            new_spikes = new_spikes[new_labels != 0]
            new_labels = new_labels[new_labels != 0]

        if randomize == True:
            subsample = create_subsamples(new_spikes, new_labels, percentage=80, nr_of_subsamples=1)
            new_spikes, new_labels = subsample[0]

        new_spikes = np.array(new_spikes)
        new_labels = np.array(new_labels)
        spikes = np.vstack((spikes, new_spikes))
        labels = np.hstack((labels, new_labels))

    # delete np.empty initialization
    return spikes[1:], labels[1:]


def create_subsamples(spikes, labels, percentage = 80, nr_of_subsamples=10):
    """
    Create a chosen number of random subsamples of the dataset each containing half of the points
    :param X: dataset
    :param nr_of_subsamples: number of subsamples to be created
    """
    subsamples = []
    for i in range(nr_of_subsamples):
        random_list = np.random.choice(range(len(spikes)), len(spikes) * percentage // 100, replace=False)
        subset_spikes = spikes[random_list, :]
        subset_labels = labels[random_list]
        subsamples.append((subset_spikes, subset_labels))
    return subsamples


def apply_fft_on_range(case, alignment, range_min, range_max):
    spikes, labels = stack_simulations_range(range_min, range_max, True, True, alignment=alignment)

    fft_real, fft_imag = apply_fft_on_data(spikes, case)

    return fft_real, fft_imag


def apply_fft_windowed_on_range(alignment, range_min, range_max, window_type):
    spikes, labels = stack_simulations_range(range_min, range_max, True, True, alignment=alignment)

    fft_real, fft_imag = apply_fft_windowed_on_data(spikes, window_type)

    return fft_real, fft_imag, labels


def apply_fft_windowed_on_sim(sim_nr, alignment, window_type):
    spikes, labels = get_dataset_simulation(simNr=sim_nr, align_to_peak=alignment)

    fft_real, fft_imag = apply_fft_windowed_on_data(spikes, window_type)

    return fft_real, fft_imag, labels


def apply_fft_on_sim(sim_nr, case, alignment):
    spikes, labels = get_dataset_simulation(simNr=sim_nr, align_to_peak=alignment)

    fft_real, fft_imag = apply_fft_on_data(spikes, case)

    return fft_real, fft_imag, labels