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


import numpy as np
from ..constants import eijk


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


def avg(x):
    return np.mean(x, axis=0)


def cumavg(data):
    return np.cumsum(data, axis=0)/np.arange(1, len(data)+1)


def movavg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    # return ret[n - 1:] / n

    # --- adaptive and keep size
    ret[:n-1] = ret[:n-1] / np.arange(1, n) * n
    return ret / n
