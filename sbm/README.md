# Space Breakdown Method
Space Breakdown Method (SBM) is a clustering algorithm that can be used to cluster low-dimensional neural data with efficiency, due to its linear complexity scaling with the data size. SBM has been published by IEEE in September 2019.

# Install:
```
pip install space-breakdown-method
```

Usage example:
```
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
```

## Paper Abstract
Overlapping clusters and different density clusters are recurrent phenomena of neuronal datasets, because of how neurons fire. We propose a clustering method that is able to identify clusters of arbitrary shapes, having different densities, and potentially overlapped. The Space Breakdown Method (SBM) divides the space into chunks of equal sizes. Based on the number of points inside the chunk, cluster centers are found and expanded. Even if we consider the particularities of neuronal data in designing the algorithm – not all data points need to be clustered, and the data space has a relatively low dimensionality – it can be applied successfully to other domains involving overlapping and different density clusters as well. The experiments performed on benchmark synthetic data show that the proposed approach has similar or better results than two well-known clustering algorithms. 

## Setup
The data used in this study can be downloaded from: https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO. The simulated data has been created by the Department of Engineering, University of Leicester UK.

The paths to the data folder on your local workstation need to be set from the 'constants.py' file (DATA_FOLDER_PATH, SIM_DATA_FOLDER_PATH, REAL_DATA_FOLDER_PATH).


# Improved Space Breakdown Method
The Improved Space Breakdown Method (ISBM) has been published in Frontiers in Computational Neuroscience: 
https://www.frontiersin.org/articles/10.3389/fncom.2023.1019637/full

The algorithm has been improved since its publishing by modifying the underlying data structure from an ndarray to a graph. Another improvement, added later to the algorithm is an adaptive Partitioning Number, influenced by the variance of each feature. This shall improve the complexity of the algorithm a bit and will allow the use of the algorithm on datasets of higher dimensions.

# Time-Frequency Breakdown Method
The Time-Frequency Breakdown Method (TFBM) has been published in Frontiers in Human Neuroscience:
https://www.frontiersin.org/articles/10.3389/fnhum.2023.1112415/full

TFBM is a method, based on SBM, developed for the detection of brain oscillations in time-frequency representations (such as spectrograms obtained from the Fourier Transform). 



# Citations
## SBM
We would appreciate it, if you cite the paper when you use this work for the original SBM algorithm:

- For Plain Text:
```
E. Ardelean, A. Stanciu, M. Dinsoreanu, R. Potolea, C. Lemnaru and V. V. Moca, "Space Breakdown Method A new approach for density-based clustering," 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP), 2019, pp. 419-425, doi: 10.1109/ICCP48234.2019.8959795.
```

- BibTex:
```
@INPROCEEDINGS{8959795,
  author={Ardelean, Eugen-Richard and Stanciu, Alexander and Dinsoreanu, Mihaela and Potolea, Rodica and Lemnaru, Camelia and Moca, Vasile Vlad},
  booktitle={2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP)}, 
  title={Space Breakdown Method A new approach for density-based clustering}, 
  year={2019},
  volume={},
  number={},
  pages={419-425},
  doi={10.1109/ICCP48234.2019.8959795}}
```
## ISBM
We would appreciate it, if you cite the paper when you use this work for the ISBM (improved SBM) algorithm:

- For Plain Text:
```
E.-R. Ardelean, A.-M. Ichim, M. Dînşoreanu, and R. C. Mureşan, “Improved space breakdown method – A robust clustering technique for spike sorting,” Frontiers in Computational Neuroscience, vol. 17, 2023, doi: 10.3389/fncom.2023.1019637.
```

- BibTex:
```
@ARTICLE{10.3389/fncom.2023.1019637,
AUTHOR={Ardelean, Eugen-Richard and Ichim, Ana-Maria and Dînşoreanu, Mihaela and Mureşan, Raul Cristian},   
TITLE={Improved space breakdown method – A robust clustering technique for spike sorting},      
JOURNAL={Frontiers in Computational Neuroscience},      
VOLUME={17},           
YEAR={2023},      
URL={https://www.frontiersin.org/articles/10.3389/fncom.2023.1019637},       
DOI={10.3389/fncom.2023.1019637}      
}
```

## TFBM
We would appreciate it, if you cite the paper when you use this work for the TFBM algorithm:

- For Plain Text:
```
E.-R. Ardelean, H. Bârzan, A.-M. Ichim, R. C. Mureşan, and V. V. Moca, “Sharp detection of oscillation packets in rich time-frequency representations of neural signals,” Frontiers in Human Neuroscience, vol. 17, 2023, doi: 10.3389/fnhum.2023.1112415.
```

- BibTex:
```
@ARTICLE{10.3389/fnhum.2023.1112415,
AUTHOR={Ardelean, Eugen-Richard and Bârzan, Harald and Ichim, Ana-Maria and Mureşan, Raul Cristian and Moca, Vasile Vlad},   
TITLE={Sharp detection of oscillation packets in rich time-frequency representations of neural signals},      
JOURNAL={Frontiers in Human Neuroscience},      
VOLUME={17},           
YEAR={2023},      
URL={https://www.frontiersin.org/articles/10.3389/fnhum.2023.1112415},       
DOI={10.3389/fnhum.2023.1112415}
}
```

# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)
