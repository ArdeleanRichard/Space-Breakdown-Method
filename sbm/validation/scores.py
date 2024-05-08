import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)
