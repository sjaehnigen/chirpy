# ChirPy

[*ChirPy*](https://hartree.chimie.ens.fr/sjaehnigen/chirpy) (_chiral python_) is a python package for chirality, dynamics, and molecular vibrations.

(... work in progress)


Main features:
- Computation of vibrational circular dichroism (VCD) from molecular dynamics (MD) and time-correlation functions (TCF)
- Application of the periodic gauge for magnetic moments and crystal symmetry in solid-state  
- Interpretation, processing, and creation of molecular topologies and supercells
- Analysis and visualisation of MD results 
    

Furthermore:
- Scientific visualisation
- Interfacing with VMD
- Processing of volumetric data and vector fields


## Installation 
Copy or clone the *ChirPy* repository into a local directory. Open a bash terminal and change directory to the local copy of the repository.

Make sure you have the following packages installed:
- `python` >= 3.10
- `pip` >= 22.3

Optional:
- `packmol` >= 20.0

Or use [Anaconda](https://anaconda.org) to create a *chirpy* environment from the `conda_env.yml` file (recommended):

`conda env create -f conda_env.yml`

In the parent directory, run:

`pip install .`

*ChirPy* has now been installed and can be imported within python:

`import chirpy as cp`

Check that you installation is correct by running from a bash terminal the test suite in `tests/`:

`python run_tests.py`

or

`./run_tests.py`

(optional arguments: --verbose, --scripts)

Thank you for reporting bugs and issues to the [developers](https://hartree.chimie.ens.fr/sjaehnigen/chirpy/-/blob/master/AUTHORS.txt).

## Examples
Workable binaries can be found in the folder `scripts/` with some pre-implemented *ChirPy* features. Make sure you add this folder to PATH.

Available jupyter notebooks and data sets:
- [Computation of Solid-State Vibrational Circular Dichroism in the Periodic Gauge](https://doi.org/10.5281/zenodo.4776906)
- 

(... under construction)

## References
1. [S. Jähnigen, A. Zehnacker, and R. Vuilleumier; Computation of Solid-State Vibrational Circular Dichroism in the Periodic Gauge, *J. Phys. Chem. Lett.*, **2021**, *12* (30), 7213-7220.](https://doi.org/10.1021/acs.jpclett.1c01682)
2. [S. Jähnigen, D. Sebastiani, R. Vuilleumier; The important role of non-covalent interactions
for the vibrational circular dichroism of lactic acid
in aqueous solution, *Phys. Chem. Chem. Phys.*, **2021**, *23*, 17232.](https://doi.org/10.1039/d1cp03106f)
