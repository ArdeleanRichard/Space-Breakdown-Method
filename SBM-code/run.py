import numpy as np
from functions import SBM
from functions import SBM_graph
import functions.dataset as ds
import functions.scatter_plot as sp
import matplotlib.pyplot as plt


def run_sbm():
    # data, y = ds.load_real_data()
    data, y = ds.load_synthetic_data(3)

    pn=25
    labels = SBM.best(data, pn, ccThreshold=5, version=2)

    sp.plot('GT' + str(len(data)), data, y, marker='o')
    sp.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')

    plt.show()


def run_sbm_graph():
    # data, y = ds.load_real_data()
    data, y = ds.load_synthetic_data(3)
    pn = 25

    labels = SBM_graph.SBM(data, pn, ccThreshold=5, version=2)

    sp.plot('GT' + str(len(data)), data, labels, marker='o')
    sp.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')

    plt.show()


if __name__ == '__main__':
    run_sbm_graph()

