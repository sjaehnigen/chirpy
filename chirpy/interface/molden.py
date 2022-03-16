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
from .. import constants


def read_moldenvib_file(filename):
    f = open(filename, 'r')
    inbuffer = f.readlines()
    f.close()
    if inbuffer[0].strip() != '[Molden Format]':
        raise Exception('No Molden format?!')
    if inbuffer[1].strip() != '[FREQ]':
        raise Exception('No Frequencies?!')
    freqs = list()
    for line in inbuffer[2:]:
        if '[FR-COORD]' in line:
            break
        else:
            freqs.append(float(line.strip()))
    freqs = np.array(freqs)
    n_modes = freqs.shape[0]
    coords = list()
    symbols = list()
    for line in inbuffer[n_modes+3:]:
        if '[FR-NORM-COORD]' in line:
            break
        else:
            tmp = line.split()
            symbols.append(tmp[0])
            coords.append([float(e) for e in tmp[1:]])
    coords_aa = np.array(coords)*constants.l_au2aa
    n_atoms = len(symbols)
    vib_data = inbuffer[n_modes+n_atoms+4:]
    modes = np.zeros((n_modes, 3*n_atoms))
    for mode in range(n_modes):
        tmp = ''.join(vib_data[mode*(n_atoms+1):(mode+1)*(n_atoms+1)][1:])
        modes[mode] = np.array([float(e.strip())
                                for e in tmp.replace('\n', ' ').split()])
    return symbols, coords_aa, freqs, modes


def write_moldenvib_file(filename, symbols, coords_aa, freqs, modes):
    n_modes = freqs.shape[0]
    n_atoms = len(symbols)
    coords = coords_aa*constants.l_aa2au
    format = '%15f'*3
    format += '\n'
    f = open(filename, 'w')
    f.write(' [Molden Format]\n [FREQ]\n')
    for i in range(n_modes):
        f.write('%16f\n' % freqs[i])
    f.write(' [FR-COORD]\n')
    for i in range(n_atoms):
        f.write((' %s '+format) % tuple([symbols[i]]+[c for c in coords[i]]))
    f.write(' [FR-NORM-COORD]\n')
    for mode in range(n_modes):
        f.write(' vibration      %i\n' % (mode+1))
        f.write(n_atoms*format % tuple([c for c in modes[mode].flatten()]))
