import matplotlib.pyplot as plt
from sklearn import datasets

from load_data import load_atoms_synthetic_data
from sbm import SBM
from sbm import ISBM
from sbm import TFBM
from sbm import scatter_plot as sp

data, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=[1.0, 2.5, 0.5], random_state=170)
pn = 15
sbm = SBM(data, pn, threshold=5)
sbm.fit()
sp.plot('GT' + str(len(data)), data, y, marker='o')
sp.plot_grid('SBM(PN=10) on Sim4', data, pn, sbm.labels, marker='o', adaptivePN=False)
plt.show()

pn = 15
isbm = ISBM(data, pn, threshold=5, adaptive=True)
isbm.fit()
sp.plot('GT' + str(len(data)), data, y, marker='o')
sp.plot_grid('ISBM(PN=25) on Sim4', data, pn, isbm.labels, marker='o', adaptivePN=True)

plt.show()

data, spectrumData = load_atoms_synthetic_data()
tfbm = TFBM(data.T, threshold="auto", merge=True, aspect_ratio=1, merge_factor=15)
tfbm.fit(verbose=True, timer=True)

tfbm.plot_result("TFBM", data, tfbm.merged_labels_data.T, tfbm.packet_infos)


