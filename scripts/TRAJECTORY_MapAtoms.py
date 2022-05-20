#!/usr/bin/env python
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


import argparse
from chirpy.classes import system


def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser = argparse.ArgumentParser(
            description="Map and reorder atoms of equal molecules.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn1", help="file 1 (xy,pdb,xvibs,...)")
    parser.add_argument("fn2", help="file 2 (xy,pdb,xvibs,...)")
    parser.add_argument(
            "--sort",
            action='store_true',
            help="Return sorted content of file 2 instead of printing.",
            default=False,
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                  angstrom/degree. If None try guessing from files.",
            default=None,
            type=float,
            )

    args = parser.parse_args()
    fn1 = args.fn1
    fn2 = args.fn2
    mol1 = system.Supercell(fn1)
    mol2 = system.Supercell(fn2)

    for _fr, _fr_b in zip(mol1.XYZ, mol2.XYZ):
        print('Frame:', _fr)
        assign = mol1.XYZ.map_frame(mol1.XYZ, mol2.XYZ,
                                    cell_aa_deg=args.cell_aa_deg)

        mol2.sort_atoms(assign)
        if args.sort:
            mol2.write_frame('sorted_frame-%06d_file2.' % _fr
                             + mol2.XYZ._fmt)

        else:
            outbuf = ['%35d -------> %3d' % (i, j)
                      for i, j in enumerate(assign)]
            print('%35s          %s' % (fn1, fn2))
            print('\n'.join(outbuf))


if(__name__ == "__main__"):
    main()
