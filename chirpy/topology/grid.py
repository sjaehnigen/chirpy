# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2020-2022, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------


import numpy as np
import warnings

from .mapping import distance_pbc
from ..config import ChirPyWarning as _ChirPyWarning

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
                      _ChirPyWarning, stacklevel=2)
    _N = 1 / (np.pi * gamma)
    _E = 1 / (1 + (r / gamma)**2)

    return _N * _E


def _lorentzian_std(r, width, dim=1):
    '''Convolution with a standardised Lorentzian function
       with maximum value at 1 and FWHM=width.'''

    _x = r / (width/2)

    return 1 / (1 + _x**2)


def regularisation(p, grid, *args, mode='gaussian', cell_aa_deg=None):
    '''Regularisation of singularities on a grid.
       Default mode uses Gaussian smear function.
       Requires *args according to chosen function.
       Expects positions p of shape (N [,dim}) with N being
       the number of points and
       pos_grid of shape ([dim,] X, [Y, Z, ...]).
       Explicit dim axis can be omitted for dim=1.
       '''
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

    if cell_aa_deg is not None:
        cell_aa_deg = cell_aa_deg[_slc]

    return np.array(
            [_F(np.linalg.norm(
                distance_pbc(
                        _p[_slc],
                        grid,
                        cell=cell_aa_deg
                        ),
                axis=0
                ),
                *args,
                dim=dim)
             for _p in p]
            )
