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
import warnings as _warnings

from ..config import ChirPyWarning as _ChirPyWarning
from .. import constants


def xvibsReader(fn, au=False, mw=False):
    '''Read an XVIBS file containing Cartesian displacements in angstrom.
       au=True/mw=True change convention by expecting atomic units and/or
       mass-weighted displacements, respectively
       '''
    with open(fn, 'r') as f:
        inbuffer = f.read()
    pos_natoms = inbuffer.index('NATOMS')
    pos_coords = inbuffer.index('COORDINATES')
    pos_freqs = inbuffer.index('FREQUENCIES')
    pos_modes = inbuffer.index('MODES')
    pos_end = inbuffer.index('&END')

    n_atoms = int(inbuffer[pos_natoms+7:pos_coords].strip())
    numbers = list()
    pos_aa = list()

    for line in inbuffer[pos_coords+12:pos_freqs].strip().split('\n'):
        tmp = line.split()
        numbers.append(int(tmp[0]))
        pos_aa.append([float(e) for e in tmp[1:4]])
    pos_aa = np.array(pos_aa)
    sec_freqs = inbuffer[pos_freqs+12:pos_modes].strip().split('\n')
    n_modes = int(sec_freqs[0])
    freqs_cgs = list()
    for line in sec_freqs[1:]:
        freqs_cgs.append(float(line))
    sec_modes = inbuffer[pos_modes+5:pos_end].strip().split('\n')
    modes = np.zeros((n_modes*n_atoms, 3))
    for i, line in enumerate(sec_modes):
        tmp = line.split()
        modes[i] = np.array([float(e) for e in tmp])
    modes = modes.reshape((n_modes, n_atoms, 3))

    if mw:
        _warnings.warn('Convention violation: '
                       'Assuming mass-weighted coordinates.',
                       _ChirPyWarning,
                       stacklevel=2)
        masses_amu = constants.numbers_to_masses(numbers)
        modes /= np.sqrt(masses_amu)[None, :, None]

    if au:
        _warnings.warn('Convention violation: '
                       'Assuming atomic unit coordinates.',
                       _ChirPyWarning,
                       stacklevel=2)
        pos_aa *= constants.l_au2aa

    return n_atoms, numbers, pos_aa, n_modes, freqs_cgs, modes
