# Neuromorphic Computing Class

Welcome to the CSCE 790 Neuromorphic Computing @ UofSC!

This repository is meant to guide students to learn how to use different SNN tools and packages including:

- [SNNToolbox](https://snntoolbox.readthedocs.io/en/latest/)

Examples of a couple of simple networks are included:

- LeNet-5
- VGG9
- AlexNet



Note: These networks have had some modifications to the particular layers versus the originals as there are some layers which are not supported for conversion techniques such as SNNToolbox.

***
## Dependencies
List of python dependencies needed to run the code in this repo:
- numpy
- matplotlib
- pandas (optional)
- tensorflow
- tensorflow-datasets
- snntoolbox

For specific versions of packages, please refer to the conda environment files (`*.yml`).

To install these packages you can use:

- `conda install <package_name>`  
OR  
- `pip install <package_name>`  

Using conda is encouraged, specifically [Miniforge3](https://github.com/conda-forge/miniforge).

***
## A Note on Python Virtual Environments
This repository contains a virtual environment for each tutorial or separate package. The main reason for this is that some (not all) of the packages have some dependency conflicts. Thus a conda environment file is provided for each of the tutorials.

The virtual environments were created using [Miniforge3](https://github.com/conda-forge/miniforge) as opposed to Anaconda.
***