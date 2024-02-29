# Automatic Prediction of Peak Optical Absorption Wavelengths in Molecules using Convolutional Neural Networks

All source code and images are associated with the paper:

"Automatic Prediction of Peak Optical Absorption Wavelengths in Molecules using Convolutional Neural Networks" 

[J. Chem. Inf. Model (2024)](https://doi.org/10.1021/acs.jcim.3c01792)

By S. G. Jung, G. Jung & J. M. Cole



## Introduction

The description of each file is summarized below:

*(i) smile_descriptors.py*

Script to generate descriptor features based on SMILES of chemical molecules & solvents. 

*(ii) smiles_feature_matrix.py*

Script to generate two-dimensional feature matrix (i.e. 2D image) of the chemical molecules & solvents based on their SMILES respresentation.

*(iii) deep_convolutional_representation.py*

Script to create and train deep residual convolutional neural networks, taking the 2D feature matrices as input.

*(iv) GBFS.py*

Script to perform gradient boosted feature selection, generate feature ranking, and carry out recursive feature selection. See "Gradient Boosted and Statistical Feature Selection" in [https://github.com/Songyosk/GBSFS4MPP](https://github.com/Songyosk/GBFS4MPPML).

*(v) Multicollinearity_reduction.py*

Script to perform multicollinearity reduction, which includes correlation analysis and hierarchical clustering analysis. Correlation and linkage thresholds are defined to elminate features. 

*(vi) optimization.py*

Script to perform Bayesian optimization, which determines the architecture of the predictive model based on a defined hyperparameter space.  

*(vii) evaluate_model.py*

Script to evaluate models by returning performance metrics and plots. 



## Model Architecture
The overview of the project pipeline:
Figure 1
![F1](Figures/UVVISPipeline.PNG)

The Deep-CNN architecture:
Figure 2
![F2](Figures/deepCNN.PNG)

The GBFS pipeline can be found in: [https://github.com/Songyosk/GBSFS4MPP](https://github.com/Songyosk/GBFS4MPPML)



## Results
The final results of the multi-fidelity prediction of the optical peaks are shown below:

![F3](Figures/GBFS_experimental_and_dft_data_random_split.png)

Figure 3: Multi-Fidelity Prediction of the Optical Peaks (Random Split)

![F4](Figures/GBFS_experimental_and_dft_data_scaffold_split.png)

Figure 4: Multi-Fidelity Prediction of the Optical Peaks (Scaffold Split)



## Acknowledgements
J.M.C. conceived the overarching project. The study was designed by S.G.J. and J.M.C. S.G.J. created the workflow, designed the CNN architecture, performed data pre-processing, featurization, hyperparameter optimization, and analysed the data under the supervision of J.M.C. G.J. assisted with the design of the CNN architecture and contributed to the hyperparameter optimization. S.G.J. drafted the manuscript with the assistance from J.M.C. The final manuscript was read and approved by all authors.

J.M.C. is grateful for the BASF/Royal Academy of Engineering Research Chair in Data-Driven Molecular Engineering of Functional Materials, which is partly sponsored by the Science and Technology Facilities Council (STFC) via the ISIS Neutron and Muon Source; this Chair also supports a PhD studentship (for S.G.J.). STFC is also thanked for a PhD studentship that is sponsored by its Scientific Computing Department (for G.J.).


## 🔗 Links
[![portfolio](https://img.shields.io/badge/Research_group-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://www.mole.phy.cam.ac.uk/)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
