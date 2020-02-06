#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np
import warnings

from .mapping import distance_pbc

# def _voigt():
#    pass

def _gaussian(r, sigma, dim=1):
    '''Convolution with a normalised Gaussian function. Integrates to one.'''
    _N = 1 / (2.*np.pi * sigma**2) ** (dim/2)
    _E = -r**2 / (2 * sigma**2)

    return _N * np.exp(_E)


def _gaussian_std(r, width, dim=1):
    '''Convolution with a standardised Gaussian function
       with maximum value at 1 and FWHM=width.'''

    _x = r / (width/2)

    return np.exp(-np.log(2) * _x**2)


def _lorentzian(r, gamma, dim=1):
    '''Convolution with a normalised Lorentzian function. Integrates to one.'''

    if dim != 1:
        warnings.warn("Lorentzian distribution not normalised for dim > 1!",
                      stacklevel=2)
    _N = 1 / (np.pi * gamma)
    _E = 1 / (1 + (r / gamma)**2)

    return _N * _E


def _lorentzian_std(r, width, dim=1):
    '''Convolution with a standardised Lorentzian function
       with maximum value at 1 and FWHM=width.'''

    _x = r / (width/2)

    return 1 / (1 + _x**2)


def regularisation(p, grid, *args, **kwargs):
    '''Regularisation of singularities on a grid.
       Default mode uses Gaussian smear function.
       Expects positions p of shape (N [,dim}) with N being
       the number of points and
       pos_grid of shape ([dim,] X, [Y, Z, ...]).
       Explicit dim axis can be omitted for dim=1.
       '''
    mode = kwargs.pop("mode", "gaussian")
    if mode == 'gaussian':
        _F = _gaussian
    elif mode == 'lorentzian':
        _F = _lorentzian
    elif mode == 'gaussian_std':
        _F = _gaussian_std
    elif mode == 'lorentzian_std':
        _F = _lorentzian_std
    else:
        raise NotImplementedError('Please use Gaussian or Lorentzian mode!')

    if len(grid.shape) == 1:
        dim = 1
        if len(p.shape) == 1:
            p = p.reshape((p.shape[0], dim))
    else:
        dim, *G = grid.shape
        if dim != len(G):
            raise TypeError('Given grid in wrong shape!')
        if dim != p.shape[1]:
            raise TypeError('Given positions in wrong shape!')

    _slc = (slice(None),) + dim * (None,)

    return np.array(
            [_F(np.linalg.norm(
                distance_pbc(
                        _p[_slc],
                        grid,
                        **kwargs
                        ),
                axis=0
                ),
                *args,
                dim=dim)
             for _p in p]
            )
