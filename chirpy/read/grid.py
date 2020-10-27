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
from itertools import islice
from concurrent_iterator.process import Producer
from .generators import _reader, _open
from ..physics import constants


def _cube(frame, **kwargs):
    '''Kernel for processing cube frame.'''
    comments = (next(frame).strip(), next(frame).strip())

    _cellinfo = list(zip(*[_l.strip().split() for _l in islice(frame, 4)]))
    cell_vec_au = np.zeros((3, 3))
    origin_au = np.zeros((3,))
    n_atoms, n_x, n_y, n_z = map(int, _cellinfo[0])
    origin_au[0], *cell_vec_au[:, 0] = map(float, _cellinfo[1])
    origin_au[1], *cell_vec_au[:, 1] = map(float, _cellinfo[2])
    origin_au[2], *cell_vec_au[:, 2] = map(float, _cellinfo[3])

    _atominfo = list(zip(*[_l.strip().split()
                           for _l in islice(frame, abs(n_atoms))]))
    pos_au = np.zeros((abs(n_atoms), 3))
    numbers = tuple(map(int, _atominfo[0]))
    # dummy = map(int, _atominfo[1])
    pos_au[:, 0] = list(map(float, _atominfo[2]))
    pos_au[:, 1] = list(map(float, _atominfo[3]))
    pos_au[:, 2] = list(map(float, _atominfo[4]))

    _nlines = 6 + abs(n_atoms) + (int(n_z / 6) + 1) * n_y * n_x
    if n_atoms < 0:
        next(frame)
        _nlines += 1
        n_atoms *= -1

    if kwargs.get('n_lines') != _nlines:
        raise ValueError('Inconsistent CUBE file!')

    data = []
    for _l in frame:
        data.extend(_l.strip().split())
    try:
        data = np.array(data).reshape(n_x, n_y, n_z).astype(float)
    except ValueError:
        raise ValueError('Tried to read broken or incomplete file!')

    return data, origin_au*constants.l_au2aa, cell_vec_au*constants.l_au2aa, \
        pos_au*constants.l_au2aa, numbers, comments


def cubeIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _cube

    with _open(FN, 'r', **kwargs) as _f:
        _f.readline()
        _f.readline()
        _natoms = int(_f.readline().strip().split()[0])
        _nx = int(_f.readline().strip().split()[0])
        _ny = int(_f.readline().strip().split()[0])
        _nz = int(_f.readline().strip().split()[0])
        _nlines = 6 + abs(_natoms) + (int(_nz / 6) + 1) * _ny * _nx

        if _natoms < 0:
            _nlines += 1

    return Producer(_reader(FN, _nlines, _kernel, **kwargs),
                    maxsize=20, chunksize=4)


def cubeReader(FN, **kwargs):
    '''Read complete XYZ file at once'''
    data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
        zip(*cubeIterator(FN, **kwargs))

    return np.array(data), origin_aa[0], cell_vec_aa[0],\
        np.array(pos_aa), numbers[0], list(comments)
