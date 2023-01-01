import numpy as np


def get_valid_neighbours(point, shape):
    neighbours = get_neighbours(point)

    neighbours = validate_neighbours(neighbours, shape)

    return neighbours

def get_neighbours(point):
    """
    Get all the coordinates of the neighbours of a point
    :param point: vector - the coordinates of the chunk we are looking at

    :returns neighbours: array - vector of coordinates of the neighbours
    """
    # ndim = the number of dimensions of a point=chunk
    ndim = len(point)

    # offsetIndexes gives all the possible neighbours ( (0,0)...(2,2) ) of an unknown point in n-dimensions
    offsetIndexes = np.indices((3,) * ndim).reshape(ndim, -1).T

    # np.r_ does row-wise merging (basically concatenate), this instructions is equivalent to offsets=np.array([-1, 0, 1]).take(offsetIndexes)
    offsets = np.r_[-1, 0, 1].take(offsetIndexes)

    # remove the point itself (0,0) from the offsets (np.any will give False only for the point that contains only 0 on all dimensions)
    offsets = offsets[np.any(offsets, axis=1)]

    # calculate the coordinates of the neighbours of the point using the offsets
    neighbours = point + offsets

    return neighbours



def validate_neighbours(neighbours, shape):
    # validate the neighbours so they do not go out of the array
    valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
    neighbours = neighbours[valid]

    return neighbours


def get_neighbours4(point):
    neighbour_offsets = np.array(
        [
            [-1, 0],
            [0, -1],
            [0, 1],
            [1, 0],
        ]
    )

    return point+neighbour_offsets




def get_neighbours_star(point, shape):
    offsets = np.array([
        [-2, 0],

        [-1, -1],
        [-1, 0],
        [-1, 1],

        [0, -2],
        [0, -1],
        [0, 0],
        [0, 1],
        [0, 2],

        [1, -1],
        [1, 0],
        [1, 1],

        [2, 0]
    ])

    neighbours = point + offsets

    neighbours = validate_neighbours(neighbours, shape)

    return neighbours



def get_neighbours_kernel(point, shape, kernel=3):
    """
    Get all the coordinates of the neighbours of a point
    :param point: vector - the coordinates of the chunk we are looking at
    :param shape: tuple - shape of the array of chunks so that we do not look outside boundaries

    :returns neighbours: array - vector of coordinates of the neighbours
    """
    # ndim = the number of dimensions of a point=chunk
    ndim = len(point)

    # offsetIndexes gives all the possible neighbours ( (0,0)...(2,2) ) of an unknown point in n-dimensions
    offsetIndexes = np.indices((kernel,) * ndim).reshape(ndim, -1).T

    # np.r_ does row-wise merging (basically concatenate), this instructions is equivalent to offsets=np.array([-1, 0, 1]).take(offsetIndexes)
    offsets=np.arange(-kernel//2+1, kernel//2+1).take(offsetIndexes)

    # remove the point itself (0,0) from the offsets (np.any will give False only for the point that contains only 0 on all dimensions)
    # offsets = offsets[np.any(offsets, axis=1)]

    # calculate the coordinates of the neighbours of the point using the offsets
    neighbours = point + offsets

    neighbours = validate_neighbours(neighbours, shape)

    return neighbours