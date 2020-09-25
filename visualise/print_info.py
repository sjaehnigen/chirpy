#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------

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
