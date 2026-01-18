# Contrasting parametric sensitivities in two global vegetation models using parameter perturbation ensembles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18285655.svg)](https://doi.org/10.5281/zenodo.18285655)

This repository contains the analysis code, notebooks, and local Python library used for the manuscript:

"Contrasting parametric sensitivities in two global vegetation models using parameter perturbation ensembles"

*Authors: Foster, Adrianna, Hawkins, Linnia R., Kennedy, Daniel, Bonan, Gordon, Fisher, Rosie, Needham, Jessica, Knox, Ryan, Koven, Charles, Wider, William, Dagon, Katherine, Lawrence, David*

Submitted to: **Journal of Advances in Modeling Earth Systems (JAMES)**

## Project Structure

| Folder/File | Description |
|---|---|
| `configs/` | Configuration files required for the analysis notebooks. | 
| `data/` | Contains an observational data file used in the study. | 
| `notebooks/` | Jupyter Notebooks: `MainAnalysis.ipynb` and `SparseGridComparison.ipynb`. | 
| `oaat_library/` | Local Python library containing core logic (`processing.py`, `plotting.py`, `utils.py`). | 
| `environment.yml` | Conda environment specification file. | 
| `pyproject.toml` | Build configuration to install `oaat_library` as a local package. | 

## Data Requirements

The analysis in this repository requires data hosted on Zenodo.

1. Download the data: [Zenodo Record 18203140](https://zenodo.org/records/18203140)
2. Placement: Download the files and update the notebooks to use that path.

## Getting Started

To ensure the notebooks run correctly, you should set up the provided Conda environment. This will install all necessary climate-science dependencies (like `xarray`, `xesmf`, and `cartopy`) and link the local `oaat_library`.

### 1. Clone the repository

```bash
git clone https://github.com/adrifoster/oaat_fates_clm
cd oaat_fates_clm
```

### 2. Create and Activate the Environment

```bash
# this will install all dependencies AND the oaat_library automatically
conda env create -f environment.yml
conda activate oaat
```

**Note: If you are adding this to an existing environment instead of creating a new one, run `pip install -e` . from the root directory.**

### 3. Download Data

Follow the instructions in the Data Requirements section above.

## Usage

After activating the environment, you can launch JupyterLab and explore the analysis:

1. `MainAnalysis.ipynb`: The primary workflow for the results presented in the paper. 
2. `SparseGridComparison.ipynb`: Comparison of the sparse grid and full grid configurations of CLM and CLM-FATEs as described in the manuscript and in the supporting information.


## Contact

For questions regarding the code or data, please contact Adrianna Foster at afoster@ucar.edu.
