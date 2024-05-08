import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

from .label_map import LABEL_COLOR_MAP



def plot_clusters(spikes, labels=None, title="", save_folder=""):
    if spikes.shape[1] == 2:
        plt.plot(title, spikes, labels)
        if save_folder != "":
            plt.savefig('./figures/' + save_folder + "/" + title)
        plt.show()
    elif spikes.shape[1] == 3:
        fig = px.scatter_3d(spikes, x=spikes[:, 0], y=spikes[:, 1], z=spikes[:, 2], color=labels.astype(str))
        fig.update_layout(title=title)
        fig.show()



def plot_spikes(spikes, step=5, title="", path='./figures/spikes_on_cluster/', save=False, show=True, ):
    """"
    Plots spikes from a simulation
    :param spikes: matrix - the list of spikes in a simulation
    :param title: string - the title of the plot
    """
    plt.figure()
    for i in range(0, len(spikes), step):
        plt.plot(np.arange(len(spikes[i])), spikes[i])
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title(title)
    if save:
        plt.savefig(path+title)
    if show:
        plt.show()


def plot_spikes_by_clusters(spikes, labels, mean=True):
    for lab in np.unique(labels):
        plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        for spike in spikes[labels==lab]:
            if mean == True:
                plt.plot(spike, 'gray')
            else:
                plt.plot(spike)
        if mean == True:
            plt.plot(np.mean(spikes[labels==lab], axis=0), LABEL_COLOR_MAP[lab])
    plt.show()


