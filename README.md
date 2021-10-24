# Space Breakdown Method
Space Breakdown Method (SBM) is a clustering algorithm that can be used to cluster low-dimensional neural data with efficiency, due to its linear complexity scaling with the data size.
SBM has been published by IEEE in September 2019:
- DOI: 10.1109/ICCP48234.2019.8959795
- Conference: 2019 IEEE 15th International Conference on Intelligent Computer Communication and Processing (ICCP)


# Paper Abstract
Overlapping clusters and different density clusters are recurrent phenomena of neuronal datasets, because of how neurons fire. We propose a clustering method that is able to identify clusters of arbitrary shapes, having different densities, and potentially overlapped. The Space Breakdown Method (SBM) divides the space into chunks of equal sizes. Based on the number of points inside the chunk, cluster centers are found and expanded. Even if we consider the particularities of neuronal data in designing the algorithm – not all data points need to be clustered, and the data space has a relatively low dimensionality – it can be applied successfully to other domains involving overlapping and different density clusters as well. The experiments performed on benchmark synthetic data show that the proposed approach has similar or better results than two well-known clustering algorithms. 

SBM results in comparison with K-Means and DBSCAN:
- on a real dataset
![Real Data](/images/real_data.PNG?raw=true)
- on a simulated dataset 
![Simulated Data](/images/simulated_data.PNG?raw=true)

# Git repository structure
The github repository contains 3 main folders:
- SBM-code : contains the code for SBM
- data : contains data files that can be loaded to test SBM on
- images: images that will be used for this readme

The SBM-code folder structure:
- run.py : already runnable code with a dataset selected
- requirements.txt : python libraries that are needed in order to run the code
- functions : folder that contains the actual SBM code

The functions folder structure:
- constants.py : file that contains some constants for different functions
- datasets.py : file containing functions used to load a dataset from the 'data' folder 
- SBM.py : file that contains the SBM algorithm in sequential or parallel versions
- SBM_functions.py : file with auxiliary functions for SBM.py
- SBM_graph.py : file that contains the improved version of SBM with the graph structure (check 'Additions' section)
- scatter_plot.py : file that contains functions for plotting datasets and results


# Citation
We would appreciate it if you cite the paper when you use this work:

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

# Additions
The algorithm has been improved since its publishing by modifying the underlying data structure from an ndarray to a graph. The following image show the improvement of a simple example from 25 cells in the ndarray to only 22 nodes in the graph.
![SBM structures](/images/sbm_structs.PNG?raw=true)

Another improvement, added later to the algorithm is an adaptive Partitioning Number, influenced by the variance of each feature. This shall improve the complexity of the algorithm a bit and will allow the use of the algorithm on datasets of higher dimensions.
![Improvement](/images/sbm_improved.PNG?raw=true)

# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)