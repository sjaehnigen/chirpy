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

from ..topology.mapping import detect_lattice, get_cell_vec


def print_header(obj):
    print('')
    print(77 * '–')
    print('%-12s' % obj.__class__.__name__)
    print(77 * '–')
    print('')


def print_cell(obj):
    print(77 * '–')
    print('CELL ' + ' '.join(map('{:10.5f}'.format, obj.cell_aa_deg)))
    print(f'{detect_lattice(obj.cell_aa_deg)}'.upper())
    print(77 * '-')
    cell_vec_aa = get_cell_vec(obj.cell_aa_deg)
    print(' A   ' + ' '.join(map('{:10.5f}'.format, cell_vec_aa[0])))
    print(' B   ' + ' '.join(map('{:10.5f}'.format, cell_vec_aa[1])))
    print(' C   ' + ' '.join(map('{:10.5f}'.format, cell_vec_aa[2])))
