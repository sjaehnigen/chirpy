#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------

import numpy as _np

from . import _CORE
from .volume import ScalarField as _ScalarField


class Domain3D(_CORE):
    '''Contains arrays of positions in a grid with assigned (scalar) values.
       The object can be expanded into a full grid representation (see volume
       class)'''

    def __init__(self,  shape,  indices,  weights,  **kwargs):
        self.grid_shape = shape
        self.indices = indices
        self.weights = weights

    def map_vector(self, v3):
        n_x, n_y, n_z = self.grid_shape
        v3_field = _np.zeros((3, n_x, n_y, n_z))
        ind = self.indices
        v3_field[:, ind[0], ind[1], ind[2]] = self.weights[None, :]*v3[:, None]
        return v3_field

    def integrate_volume(self, f):
        # return simps(f(self.indices)*self.weights)
        return _np.sum(f(self.indices)*self.weights, axis=0)

    def expand(self):
        data = _np.zeros(self.grid_shape)
        data[self.indices] = self.weights
        return data

    def write(self, fn, **kwargs):
        _ScalarField.from_domain(self, **vars(self)).write(fn, **kwargs)
