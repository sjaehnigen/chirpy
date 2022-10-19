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
- Handling of volumetric data and vector fields


## Installation 
Copy or clone the *ChirPy* repository into a local folder. 
To integrate this code into your python environment, please update your `$PYTHONPATH` or adjust and source the file `paths.conf`.

It is highly recommended to use [Anaconda](https://anaconda.org) to create a *ChirPy* environment from the `conda_env.yml` file.

Tests can be run in the folder `tests/`

`python run.py`

or in bash terminal

`./run.py`


Thank you for reporting bugs and issues to the [developers](https://hartree.chimie.ens.fr/sjaehnigen/chirpy/-/blob/master/AUTHORS.txt).

## Examples
Workable binaries can be found in the folder `scripts/` with some pre-implemented *ChirPy* features.

Available jupyter notebooks and data sets:
- [Computation of Solid-State Vibrational Circular Dichroism in the Periodic Gauge](https://doi.org/10.5281/zenodo.4776906)
- 

(... under construction)

## References
1. [S. Jähnigen, A. Zehnacker, and R. Vuilleumier; Computation of Solid-State Vibrational Circular Dichroism in the Periodic Gauge, *J. Phys. Chem. Lett.*, **2021**, *12* (30), 7213-7220.](https://doi.org/10.1021/acs.jpclett.1c01682)
2. [S. Jähnigen, D. Sebastiani, R. Vuilleumier; The important role of non-covalent interactions
for the vibrational circular dichroism of lactic acid
in aqueous solution, *Phys. Chem. Chem. Phys.*, **2021**, *23*, 17232.](https://doi.org/10.1039/d1cp03106f)
