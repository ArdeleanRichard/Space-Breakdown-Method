import copy
import time
import random

import numpy as np

from scipy.ndimage import maximum_filter
from sklearn.preprocessing import LabelEncoder

from collections import deque

from common.distance import euclidean_point_distance_scale_fast
from common.neighbourhood import get_valid_neighbours8
from preprocess.data_scaling import normalize_data_min_max


class PacketInfo:
    def __init__(self):
        """
        center_coords: 2D point indicating the XY coordinates of the current center
        parent_coords: 2D point indicating the XY coordinates of the parent cluster
        cluster_points: list of 2D points that determine the oscillation packet
        contour_points: list of 2D points that determine only the contour of the oscillation packet
        conflict_dict: dictionary that represents all the conflicts of the current cluster - with keys (coordinates of centres), values all the points of conflict
        peak: value of the centre in the image
        prominence: calculate as the difference between the peak and the maximum value of the contour
        start_label: label assigned at the expansion of the oscillation packets
        finish_label: label assigned after the merging process
        """
        self.center_coords = None
        self.parent_center_coords = None
        self.packet_points = []
        self.contour_points = []
        self.conflict_dict = None
        self.peak = None
        self.actual_peak = None
        self.start_label = None

        self.prominence = None
        self.actual_prominence = None

        self.finish_label = None
        self.updated_packet_points = None
        self.updated_contour_points = None
        self.updated_conflict_dict = None

        self.packet_mat = None


class TFBM:
    def __init__(self, data, threshold="auto", aspect_ratio=1, merge=True, merge_factor=15):
        self.actual_data = np.copy(data)
        self.data = np.copy(data)
        self.conflict_mat = np.zeros_like(self.data, dtype=int)

        min_dim = min(self.data.shape[1], self.data.shape[0])
        if aspect_ratio > 1:
            self.scale = np.array([min_dim / self.data.shape[0] / aspect_ratio, min_dim / self.data.shape[1]])
        elif aspect_ratio < 1:
            self.scale = np.array([min_dim / self.data.shape[0], min_dim / self.data.shape[1] * aspect_ratio])
        elif aspect_ratio == 1:
            self.scale = np.array([min_dim / self.data.shape[0], min_dim / self.data.shape[1]])


        self.data = normalize_data_min_max(self.data) * 100

        hist, bin_edges = np.histogram(data, bins=100)
        cumsum = np.cumsum(hist)
        self.exp_thr = np.argwhere(cumsum > 90 / 100 * cumsum[-1])[0][0]

        if threshold is "auto":
            # The cumulative distribution (bottom panel) is used to set the threshold
            # to cover 90% of all power values,
            # which translate into less than 10% of the maximum power.
            hist, bin_edges = np.histogram(data, bins=100)
            cumsum = np.cumsum(hist)
            self.threshold = np.argwhere(cumsum > 90 / 100 * cumsum[-1])[0][0]
            self.exp_thr = self.threshold
            self.threshold_type = "auto"
        else:
            self.threshold = threshold / 100 * np.amax(self.data)
            self.threshold_type = "manual"

        self.merge = merge
        self.merge_factor = merge_factor

        self.packet_infos = []

        self.labels_data = np.zeros_like(self.data, dtype=int)
        self.merged_labels_data = np.zeros_like(self.data, dtype=int)

    def fit(self, verbose=False, timer=False):
        if verbose == True:
            print()
            print("------- TFBM starts --------")
            print(f"Threshold set at {self.threshold}")

        start = time.time()
        self.find_packet_centers_no_neighbours()
        if verbose == True:
            print(f"Found {len(self.packet_infos)} packet centers with "
                  f"{'automatically' if self.threshold_type is 'auto' else 'manually'} "
                  f"set threshold at {self.threshold}")
        if timer == True:
            print(f"PCS timer: {time.time() - start:.5f}s")

        start = time.time()
        self.expand_all_packets()
        if timer == True:
            print(f"EXP timer: {time.time() - start:.5f}s")
        if verbose == True:
            print(f"{len(np.unique(self.labels_data))} packets expanded, containing {len(self.packet_infos)} local maxima")

        if self.merge == True:
            start = time.time()
            self.merge_labels()
            if timer == True:
                print(f"MERGE timer: {time.time() - start:.5f}s")
            if verbose == True:
                print(f"Merged into {len(np.unique(self.merged_labels_data))} packets found, containing {len(self.packet_infos)} local maxima")

        self.reencode_labels()

        if verbose == True:
            print("------- TFBM stops --------")
            print()

    def find_packet_centers_no_neighbours(self):
        """
        Search through the matrix of chunks to find the cluster centers
        """

        packet_centers = np.argwhere((maximum_filter(self.data, size=3) == self.data) & (self.data > self.threshold))

        # create packet_infos
        self.create_packet_infos(packet_centers)

    def expand_all_packets(self):
        for current_id, packet_info in enumerate(self.packet_infos):
            self.expand_packet_center(current_id)

        self.og_labels = np.copy(self.labels_data)
        self.packet_conflict_solver()

        self.merged_labels_data = np.copy(self.labels_data)
        self.calculate_contours_and_conflicts()
        self.calculate_prominences()

    def expand_packet_center(self, current_id):  # TODO
        """
        Expansion
        :param current_id: integer - the id of the current cluster center

        """
        start = tuple(self.packet_infos[current_id].center_coords)

        current_label = current_id + 1

        self.packet_infos[current_id].parent_center_coords = None
        self.packet_infos[current_id].start_label = current_label
        self.packet_infos[current_id].finish_label = current_label
        self.packet_infos[current_id].packet_mat = np.zeros_like(self.data, dtype=bool)

        # init class and queue
        expansionQueue = deque()
        expansionQueue.append(start)
        self.packet_infos[current_id].packet_mat[start] = current_label
        self.labels_data[start] = current_label

        neighbours = get_valid_neighbours8(start, self.data.shape)
        dropoff = np.sqrt(np.sum((self.data[start] - np.amin(self.data[neighbours[:, 0], neighbours[:, 1]])) ** 2))

        while expansionQueue:
            point = expansionQueue.popleft()

            neighbours = get_valid_neighbours8(point, self.data.shape)
            dist = euclidean_point_distance_scale_fast(np.array(start), np.array(point), scale=self.scale)
            number = dropoff * dist

            for neigh_id, neighbour_coord in enumerate(neighbours):
                neighbour = tuple(neighbour_coord)

                if self.packet_infos[current_id].packet_mat[neighbour] == 0 and number <= self.data[neighbour] <= self.data[point]:
                    expansionQueue.append(neighbour)

                    if self.labels_data[neighbour] == 0:
                        self.labels_data[neighbour] = current_label
                    else:
                        self.labels_data[neighbour] = -1 # indicate conflict point
                    self.packet_infos[current_id].packet_mat[neighbour] = 1

        self.conflict_mat = self.conflict_mat + self.packet_infos[current_id].packet_mat

    def packet_conflict_solver(self):
        (conflict_x, conflict_y) = np.where(self.conflict_mat > 1)

        for x, y in zip(conflict_x, conflict_y):
            packet_ids = []
            for current_id, packet_info in enumerate(self.packet_infos):
                if self.packet_infos[current_id].packet_mat[x, y] == 1:
                    packet_ids.append(current_id)

            max = 0
            max_id = None
            for packet_id in packet_ids:
                pc = self.packet_infos[packet_id].center_coords
                dist = euclidean_point_distance_scale_fast(np.array([x, y]), np.array(pc), scale=self.scale)
                if self.data[pc] / dist > max:
                    max = self.data[pc] / dist
                    max_id = packet_id

            if max_id is not None:
                self.labels_data[x, y] = max_id + 1

        for current_id, packet_info in enumerate(self.packet_infos):
            current_label = current_id + 1

            (points_x, points_y) = np.where(self.labels_data == current_label)
            points = []
            for (x, y) in zip(points_x, points_y):
                points.append((x, y))

            self.packet_infos[current_id].packet_points = points
            self.packet_infos[current_id].updated_packet_points = copy.deepcopy(points)

    def merge_labels(self):
        for current_id in range(len(self.packet_infos) - 1, -1, -1):
            current_label = self.packet_infos[current_id].finish_label

            while True:
                conflict_center_str = None
                for conflict_center_string in list(self.packet_infos[current_id].updated_conflict_dict.keys()):
                    if self.packet_infos[current_id].updated_conflict_dict[conflict_center_string] is not None:
                        conflict_center_str = conflict_center_string
                        break

                if conflict_center_str is None:
                    break

                conflict_center = eval(conflict_center_str)

                conflict_id = self.find_info_id_by_center(conflict_center)
                conflict_label = self.packet_infos[conflict_id].finish_label

                # safety, conflict id not found, should never get here
                if conflict_id is None:
                    break

                # if no conflict exists between current and conflict packet
                if self.packet_infos[current_id].updated_conflict_dict[conflict_center_str] is None:
                    break

                # the conflict packet was already absorbed by another packet, skip it
                if self.packet_infos[current_id].parent_center_coords is not None:
                    break

                conflicts_coords = self.packet_infos[current_id].updated_conflict_dict[f"{conflict_center}"]

                max_conflict_value = 0
                for conflict_coord in conflicts_coords:
                    if self.data[tuple(conflict_coord)] > max_conflict_value:
                        max_conflict_value = self.data[tuple(conflict_coord)]


                if self.packet_infos[current_id].peak < self.packet_infos[conflict_id].peak:
                    if self.packet_infos[current_id].peak - max_conflict_value < self.merge_factor \
                            and self.packet_infos[conflict_id].peak - max_conflict_value < self.merge_factor:
                        self.merged_labels_data[self.merged_labels_data == current_label] = conflict_label
                        self.update_infos(loser_id=current_id, winner_id=conflict_id)

                self.packet_infos[current_id].updated_conflict_dict[conflict_center_str] = None

        self.calculate_merged_contours()

    def find_info_id_by_center(self, conflict_center):
        for i in range(len(self.packet_infos)):
            if self.packet_infos[i].center_coords[0] == conflict_center[0] and self.packet_infos[i].center_coords[1] == conflict_center[1]:
                return i
        return None

    def find_info_id_by_start_label(self, label):
        for i in range(len(self.packet_infos)):
            if self.packet_infos[i].start_label == label:
                return i
        return None

    def update_conflict_dict(self, winner_id, loser_id):
        winner_conflict_dict = self.packet_infos[winner_id].updated_conflict_dict
        loser_conflict_dict = self.packet_infos[loser_id].updated_conflict_dict

        winner_conflict_keys = list(winner_conflict_dict.keys())
        loser_conflict_keys = list(loser_conflict_dict.keys())

        for loser_key in loser_conflict_keys:
            if loser_key in winner_conflict_keys:
                if winner_conflict_dict[loser_key] is None:
                    continue
                else:
                    if loser_conflict_dict[loser_key] is None:
                        continue
                    else:
                        winner_conflict_dict[loser_key].extend(loser_conflict_dict[loser_key])
            else:
                winner_conflict_dict[loser_key] = loser_conflict_dict[loser_key]
            loser_conflict_dict[loser_key] = None

    def update_infos(self, loser_id, winner_id):
        self.packet_infos[loser_id].parent_center_coords = self.packet_infos[winner_id].center_coords
        self.packet_infos[loser_id].finish_label = self.packet_infos[winner_id].finish_label

        self.packet_infos[winner_id].updated_packet_points.extend(self.packet_infos[loser_id].updated_packet_points)

        # PARENT RECEIVES CONFLICTS OF KID
        self.update_conflict_dict(winner_id, loser_id)

        # winner gets himself removed from conflicts just in case the loser had him
        self.packet_infos[winner_id].updated_conflict_dict[f"{self.packet_infos[winner_id].center_coords}"] = None

        # winner gets loser removed from conflicts as it was absorbed
        self.packet_infos[winner_id].updated_conflict_dict[f"{self.packet_infos[loser_id].center_coords}"] = None

    def reencode_labels(self):
        # REENCODE LABELS
        shape = self.merged_labels_data.shape
        le = LabelEncoder()
        # ravel removes DataConversionWarning: A column-vector y was passed when a 1d array was expected.
        self.merged_labels_data = le.fit_transform(self.merged_labels_data.reshape(-1, 1).ravel())
        self.merged_labels_data = self.merged_labels_data.reshape(shape)

        unique_labels = np.unique(self.merged_labels_data)
        # shuffle labels for colouring
        label_list = list(range(1, len(unique_labels)))  # index from one because background remains the same
        random.shuffle(label_list)

        reencoded_labels = np.zeros_like(self.merged_labels_data)
        for id, label in enumerate(unique_labels[1:]):  # index from one to not modify background
            reencoded_labels[self.merged_labels_data == label] = label_list[id]

        self.merged_labels_data = reencoded_labels

    def create_packet_infos(self, packet_centers):
        # preprocess
        # create sub-lists of pairs [coordinates, peak value]
        packet_pairs = np.zeros((len(packet_centers), 3))
        packet_pairs[:, 0:2] = packet_centers
        packet_pairs[:, 2] = self.data[packet_centers[:, 0], packet_centers[:, 1]]

        # pairs composed of sub-lists of [tuple coordinates, peak value]
        # sort by peaks
        sorted_pairs = packet_pairs[packet_pairs[:, 2].argsort()][::-1]
        # print(sorted_pairs)

        self.rightest_id = np.argmax(sorted_pairs[:, 1])
        # self.rightest = (int(sorted_pairs[rightest_id, 0]),int(sorted_pairs[rightest_id, 1]))

        for pair in sorted_pairs:
            info = PacketInfo()
            info.center_coords = (int(pair[0]), int(pair[1]))
            info.peak = pair[2]
            info.actual_peak = self.actual_data[info.center_coords]
            self.packet_infos.append(info)

    def calculate_prominences(self):
        # CALCULATE PROMINENCE

        for id, info in enumerate(self.packet_infos):
            points2D = np.array(self.packet_infos[id].contour_points)

            self.packet_infos[id].prominence = self.packet_infos[id].peak - np.amax(self.data[points2D[:, 0], points2D[:, 1]])
            self.packet_infos[id].actual_prominence = self.packet_infos[id].actual_peak - np.amax(self.actual_data[points2D[:, 0], points2D[:, 1]])
            self.packet_infos[id].inv_prominence = self.packet_infos[id].peak - np.amin(self.data[points2D[:, 0], points2D[:, 1]])


    def calculate_contours_and_conflicts(self):
        # CALCULATE CONTOURS
        for id, info in enumerate(self.packet_infos):
            conflicts = {}
            contour = []
            temp_matrix = np.zeros_like(self.data)

            points2D = np.array(info.packet_points)
            temp_matrix[points2D[:, 0], points2D[:, 1]] = 1

            for point in info.packet_points:
                neighbours = get_valid_neighbours8(point, temp_matrix.shape)

                result = np.any(temp_matrix[neighbours[:, 0], neighbours[:, 1]] == 0)

                if result == True:
                    contour.append(point)

                    for neighbour in neighbours:
                        neigh_label = self.labels_data[tuple(neighbour)].astype(int)

                        if neigh_label != 0:
                            neighb_id = self.find_info_id_by_start_label(neigh_label)
                            if id != neighb_id:
                                key = f"{self.packet_infos[neighb_id].center_coords}"
                                if key in conflicts.keys():
                                    conflicts[key].append(neighbour)
                                else:
                                    conflicts[key] = []
                                    conflicts[key].append(neighbour)

            self.packet_infos[id].contour_points = contour
            self.packet_infos[id].conflict_dict = conflicts
            self.packet_infos[id].updated_conflict_dict = copy.deepcopy(conflicts)

    def calculate_merged_contours(self):
        for id, info in enumerate(self.packet_infos):
            if info.parent_center_coords is None:
                contour = []
                temp_matrix = np.zeros_like(self.labels_data)

                points2D = np.array(info.updated_packet_points)
                temp_matrix[points2D[:, 0], points2D[:, 1]] = 1

                for point in info.updated_packet_points:
                    neighbours = get_valid_neighbours8(point, temp_matrix.shape)

                    result = np.any(temp_matrix[neighbours[:, 0], neighbours[:, 1]] == 0)
                    if result == True:
                        contour.append(point)

                self.packet_infos[id].updated_contour_points = np.array(contour)
            else:
                self.packet_infos[id].updated_contour_points = np.array(copy.deepcopy(self.packet_infos[id].contour_points))


