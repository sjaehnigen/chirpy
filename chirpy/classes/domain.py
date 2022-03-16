# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2022, The ChirPy Developers.
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
# -------------------------------------------------------------------

import numpy as _np
import copy as _copy

from .core import CORE as _CORE
from .volume import ScalarField as _ScalarField
from ..snippets import extract_keys


class Domain3D(_CORE):
    '''Contains arrays of positions in a grid with assigned (scalar) values.
       The object can be expanded into a full grid representation (see volume
       class)'''

    def __init__(self,  shape,  indices,  weights,  **kwargs):
        self.grid_shape = shape
        self.indices = indices
        self.weights = weights

    def __add__(self, other):
        if self.grid_shape != other.grid_shape:
            raise ValueError('cannot combine domains of grid shape '
                             f'{self.grid_shape} and {other.grid_shape}')
        new = _copy.deepcopy(self)
        data = self.expand() + other.expand()
        new.indices = _np.where(data != 0)
        new.weights = data[new.indices]
        return new

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
        _ScalarField.from_domain(self, **extract_keys(vars(self),
                                                      origin_aa=None,
                                                      pos_aa=None,
                                                      cell_vec_aa=None,
                                                      numbers=None)
                                 ).write(fn, **kwargs)
