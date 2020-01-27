#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np
from .mapping import distance_pbc


def _gaussian(posgrid, b, sigma, cell_aa_deg=None, dim=3):
    '''Convolution with a Gaussian function. Integrates to one.'''
    _N = 1 / (2.*np.pi * sigma**2) ** (dim/2)
    _slc = (slice(None),) + dim * (None,)

    if cell_aa_deg is not None:
        _r = distance_pbc(b[_slc], posgrid, cell_aa_deg=cell_aa_deg[_slc])
    else:
        _r = posgrid - b[_slc]

    _E = -(np.linalg.norm(_r, axis=0) ** 2) / (2 * sigma**2)
    return _N * np.exp(_E)


def _lorentzian(posgrid, b, gamma, cell_aa_deg=None, dim=3):
    '''Convolution with a Lorentzian function. Integrates to one.'''
    _N = 1 / (np.pi * gamma)
    _slc = (slice(None),) + dim * (None,)

    if cell_aa_deg is not None:
        _r = distance_pbc(b[_slc], posgrid, cell_aa_deg=cell_aa_deg[_slc])
    else:
        _r = posgrid - b[_slc]

    _E = 1 / (1 + (np.linalg.norm(_r, axis=0) / gamma)**2)
    return _N * _E


def map_on_posgrid(p, posgrid, *args, **kwargs):
    '''Regularisation of singularities on a position_grid.
       Default mode uses Gaussian smear function.
       Expects positions p of shape (dim,) and
       pos_grid of shape (dim, X, [Y, Z, ...])
       (default: dim=3).
       '''
    mode = kwargs.pop("mode", "gaussian")
    if mode == 'gaussian':
        return _gaussian(posgrid, p, *args, **kwargs)
    elif mode == 'lorentzian':
        return _lorentzian(posgrid, p, *args, **kwargs)
    else:
        raise NotImplementedError('Please use Gaussian or Lorentzian mode!')
