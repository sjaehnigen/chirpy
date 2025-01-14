# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2024, The ChirPy Developers.
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

import sys
from ..topology.mapping import detect_lattice, cell_vec


def print_header(obj):
    print(f'''
{77 * '–'}
{'%-12s' % obj.__class__.__name__}
{77 * '–'}
''',
          file=sys.stderr)


def print_cell(obj):
    if getattr(obj, 'cell_aa_deg') is None:
        return
    cell_vec_aa = cell_vec(obj.cell_aa_deg)

    print(f'''
{77 * '–'}
CELL {' '.join(map('{:10.5f}'.format, obj.cell_aa_deg))}
{str(detect_lattice(obj.cell_aa_deg)).upper()}
{77 * '-'}
 A   {' '.join(map('{:10.5f}'.format, cell_vec_aa[0]))}
 B   {' '.join(map('{:10.5f}'.format, cell_vec_aa[1]))}
 C   {' '.join(map('{:10.5f}'.format, cell_vec_aa[2]))}
''',
          file=sys.stderr)
