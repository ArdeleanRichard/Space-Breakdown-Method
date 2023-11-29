# Space Breakdown Method
Space Breakdown Method (SBM) is a clustering algorithm that can be used to cluster low-dimensional neural data with efficiency, due to its linear complexity scaling with the data size.
SBM has been published by IEEE in September 2019:
- https://ieeexplore.ieee.org/document/8959795
- DOI: 10.1109/ICCP48234.2019.8959795
- Conference: 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP)


## Paper Abstract
Overlapping clusters and different density clusters are recurrent phenomena of neuronal datasets, because of how neurons fire. We propose a clustering method that is able to identify clusters of arbitrary shapes, having different densities, and potentially overlapped. The Space Breakdown Method (SBM) divides the space into chunks of equal sizes. Based on the number of points inside the chunk, cluster centers are found and expanded. Even if we consider the particularities of neuronal data in designing the algorithm – not all data points need to be clustered, and the data space has a relatively low dimensionality – it can be applied successfully to other domains involving overlapping and different density clusters as well. The experiments performed on benchmark synthetic data show that the proposed approach has similar or better results than two well-known clustering algorithms. 

SBM results in comparison with K-Means and DBSCAN:
- on a real dataset
![Real Data](/images/real_data.PNG?raw=true)
- on a simulated dataset 
![Simulated Data](/images/simulated_data.PNG?raw=true)

## Git repository structure
The code folder structure:
- main.py : main file to run the code from
- requirements.txt : python libraries that are needed in order to run the code
- images: images that will be used for this readme
- algorithms : folder that contains the actual SBM and ISBM code
- dataset parsing: scripts for parsing different datasets
- common/validation/visualization: utilitary functions

## Setup
The 'requirements.txt' file indicates the dependencies required for running the code. The data used in this study can be downloaded from: https://1drv.ms/u/s!AgNd2yQs3Ad0gSjeHumstkCYNcAk?e=QfGIJO. 

The paths to the data folder on your local workstation need to be set from the 'constants.py' file (DATA_FOLDER_PATH, SIM_DATA_FOLDER_PATH, REAL_DATA_FOLDER_PATH).


# Improved Space Breakdown Method
The Improved Space Breakdown Method has been published in Frontiers in Computational Neuroscience: 
https://www.frontiersin.org/articles/10.3389/fncom.2023.1019637/full

The algorithm has been improved since its publishing by modifying the underlying data structure from an ndarray to a graph. The following image show the improvement of a simple example from 25 cells in the ndarray to only 22 nodes in the graph.
![SBM structures](/images/isbm_struct.PNG?raw=true)

Another improvement, added later to the algorithm is an adaptive Partitioning Number, influenced by the variance of each feature. This shall improve the complexity of the algorithm a bit and will allow the use of the algorithm on datasets of higher dimensions.
![Improvement](/images/isbm.png?raw=true)


# Time-Frequency Breakdown Method


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
E.-R. Ardelean, A.-M. Ichim, M. Dînşoreanu, and R. C. Mureşan, “Improved space breakdown method – A robust clustering technique for spike sorting,” Frontiers in Computational Neuroscience, vol. 17, 2023
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

```

- BibTex:
```

```

# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)
