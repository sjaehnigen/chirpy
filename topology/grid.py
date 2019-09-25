#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


import numpy as np
from .symmetry import distance_pbc

def gaussian(posgrid, b, sigma, cell_aa_deg=None, dim=3):
    '''Regularisation using Gaussian function'''
    _N = 1/(2.*np.pi*sigma**2)**(dim/2)
    _slc = (slice(None),) + dim * (None,)

    #use cell_vec_au in the first instance?
    if cell_aa_deg is not None:
        _r = distance_pbc(posgrid, b[_slc], cell_aa_deg=cell_aa_deg[_slc])
    else:
        _r = posgrid - b[_slc]

    _E = -(np.linalg.norm(_r, axis=0)**2)/(2*sigma**2)
    return _N * np.exp(_E)

def map_on_posgrid(p, posgrid, sigma, cell_aa_deg=None, mode="gaussian", dim=3):
    #Do not use cell_vec_au (watch out when using cubes .. still no uniform notation)
    #cell_aa_deg should be replaced by cell_vec_au very soon
    if mode == 'gaussian':
        return gaussian(posgrid, p, sigma, cell_aa_deg=cell_aa_deg, dim= dim)
    else:
        raise NotImplementedError('Please use Gaussian mode!')

