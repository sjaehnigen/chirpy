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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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

from functools import partial
import numpy as np
import copy

from . import _CORE
from ..topology.dissection import fermi_cutoff_function
from ..topology.mapping import distance_pbc


class Sphere(_CORE):
    def __init__(self, position=None, radius=None, edge='hard', D=0.23622):
        '''Define a sphere at position, radius and edge (soft/hard).
           D=0.23622 bohr corresponds to 0.125 angstrom (soft sphere only)
           Expects positions of shape ([n_frames, ...] dim).
           Radius can be a float (costants) or an array of shape (n_frames)
           (dynamic).
           '''
#        if len(position.shape) != 2:
#            raise TypeError('Got wrong shape for sphere position!',
#                            position.shape)
        if not isinstance(radius, float):
            if isinstance(radius, np.ndarray):
                if len(radius) != len(position):
                    raise TypeError('Got wrong length for radius!', radius)
                radius = radius[:, None]
                #  radius = radius[None, :]  # ?
            else:
                raise TypeError('Expected float or numpy array for radius!')

        self.pos = position
        self.r = radius
        if edge == 'hard':
            self.edge = lambda d: (d <= self.r).astype(float)
        elif edge == 'soft':
            self.edge = partial(fermi_cutoff_function, R_cutoff=self.r, D=D)

    def clip_section_observable(self, x, pos, cell=None, inverse=False):
        '''Apply sphere on observable x using.
           Scaling is applied according to positions relative to the sphere's
           origin.

           Expects x of shape ([FR,] N, [1 ...]) and pos{itions} of shape
           ([FR, N,] 3) that have to correspond to Sphere position's shape.
           cell: [a, b, c, al, be, ga]
           BETA
           '''

        def get_d(orig, _pos):
            return np.linalg.norm(distance_pbc(orig, _pos, cell=cell), axis=-1)

        # --- never manipulate input
        _x = copy.deepcopy(x)
        keep = copy.deepcopy(x)

        if len(pos.shape) > 3:
            raise TypeError('Got wrong shape for pos!', pos.shape)
        elif pos.shape == self.pos.shape:
            _d = get_d(self.pos, pos)
        elif pos.shape[::2] == self.pos.shape:
            _d = get_d(self.pos[:, None], pos)
        elif pos.shape[::2] == self.pos.shape[::2]:
            _d = get_d(self.pos[:, None], pos[:, :, None])
        else:
            raise TypeError('Clip reference shape %s does not agree with '
                            'sphere position shape %s!' %
                            (pos.shape, self.pos.shape))

        _slc = (slice(None),) * len(_d.shape) + (None,)
        _x *= self.edge(_d)[_slc]

        if inverse:
            return keep - _x
        else:
            return _x
