import numpy as np

from .neighbourhood import get_valid_neighbours


def check_maxima(array, point):
    """
    Check whether the chunk with the coordinates "point" is bigger than all its neighbours
    :param array: matrix - an array of the values in each chunk
    :param point: vector - the coordinates of the chunk we are looking at

    :returns : boolean - whether or not it is a maxima
    """
    neighbours = get_valid_neighbours(point, np.shape(array))

    for neighbour in neighbours:
        if array[tuple(neighbour)] > array[point]:
            return False
    # fastest way but you need to convert array of arrays to array of tuples
    # neighbours = np.apply_along_axis(tuple, 0, neighbours)
    # this doesnt work because python retarded
    # print(np.all(array[neighbours] < array[point]))

    return True



def check_maxima_no_neighbour_maxim(array, point, maximas):
    """
    Check whether the chunk with the coordinates "point" is bigger than all its neighbours
    :param array: matrix - an array of the values in each chunk
    :param point: vector - the coordinates of the chunk we are looking at

    :returns : boolean - whether or not it is a maxima
    """
    neighbours = get_valid_neighbours(point, np.shape(array))

    for neighbour in neighbours:
        if array[tuple(neighbour)] > array[tuple(point)]:
            return False
        for cc in maximas:
            if tuple(neighbour) == tuple(cc):
                return False

    return True