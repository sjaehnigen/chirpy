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


import numpy as np
from ..physics.constants import eijk


def divrot(data, cell_vec):
    '''Gridded calculation of divergence and rotation of a vector field
       using gradient along the dimensions of the grid (x, y, z only in the
       case of tetragonal cells).

           data of shape 3, x, y, z
           cell_vec ... grid unit vectors (spacing)
       '''
    gradients = np.array(np.gradient(data, 1,
                                     np.linalg.norm(cell_vec[0]),
                                     np.linalg.norm(cell_vec[1]),
                                     np.linalg.norm(cell_vec[2]))[1:])
    div = gradients.trace(axis1=0, axis2=1)
    rot = np.einsum('ijk, jklmn->ilmn', eijk, gradients)

    return div, rot
