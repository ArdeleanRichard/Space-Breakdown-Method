import numpy as np

def euclidean_point_distance(pointA, pointB):
    """
    Calculates the euclidean distance between 2 points (L2 norm/distance) for n-dimensional points
    :param pointA: vector - vector containing all the dimensions of a point A
    :param pointB: vector - vector containing all the dimensions of a point B

    :returns dist: float - the distance between the 2 points
    """
    difference = np.subtract(pointA, pointB)
    squared = np.square(difference)
    dist = np.sqrt(np.sum(squared))
    return dist
