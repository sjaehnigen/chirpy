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

import sys
from ..topology.mapping import detect_lattice, get_cell_vec


def print_header(obj):
    print(f'''
{77 * '–'}
{'%-12s' % obj.__class__.__name__}
{77 * '–'}
''',
          file=sys.stderr)


def print_cell(obj):
    if not hasattr(obj, 'cell_aa_deg'):
        return
    cell_vec_aa = get_cell_vec(obj.cell_aa_deg)

    print(f'''
{77 * '–'}
CELL {' '.join(map('{:10.5f}'.format, obj.cell_aa_deg))}
{detect_lattice(obj.cell_aa_deg).upper()}
{77 * '-'}
 A   {' '.join(map('{:10.5f}'.format, cell_vec_aa[0]))}
 B   {' '.join(map('{:10.5f}'.format, cell_vec_aa[1]))}
 C   {' '.join(map('{:10.5f}'.format, cell_vec_aa[2]))}
''',
          file=sys.stderr)
