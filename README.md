# ROBUST IDENTIFIABILITY FOR SYMBOLIC RECOVERY OF DIFFERENTIAL EQUATIONS

Robust Identifiability for symbolic recovery of differntial equations provides Code for analysing the identifiablity of governing equations with noise.

## Installation
To set up this project, first clone this repository:

```bash
git clone git@github.com:HillaryHauger/robust-identifiability-symbolic-recovery.git
```
Next, create a new virtual environment (e.g. using conda), then run
```bash
cd noise-robust-identifiability-physical-law-learning
pip install -e .
````

This will install all required dependencies.

If encountered with the error 

```bash
../meson.build:48:4: ERROR: Problem encountered: OpenBLAS or Netlib BLAS/CBLAS is required on all platforms except Windows, and was not found
```
you have to install OpenBLAS manually (https://github.com/OpenMathLib/OpenBLAS), this can be done using the following commands
```bash
sudo apt update
sudo apt install libopenblas-dev
```
Rerunning the installation should work then.


## Structure

The repository is organized as follows:

`experiments/`: Contains scripts and configurations for running experiments on NR-FRanCo and NR-JRC.

`methods/`: Includes the implementation of NR-FRanCO and NR-JRC. This includes functionalities to bound the error on finite differences and the function $\frac{\sigma_n}{\sigma_1}$ as well as the calculation of the derivatives with the pysindy package.

`results/`: Stores results from running experiments in 'nr_jrc_and_nr_franco.ipynb' as a csv file and figures produced in the experiments folder.

`utils/`: Contains utility scripts and helper functions for creating the test_data and so on.
     
## Usage

The experiments can be run with jupyter notebooks in the experiments folder for NR_FRanCo and NR-JRC. The 'nr_jrc_and_nr_franco.ipynb' compares the results of both algorithms for four different PDEs and shows how many where correctly calculated.
