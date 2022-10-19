#!/usr/bin/env python
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

import argparse
from chirpy.classes import system


def main():
    '''Print spread in angstrom.'''
    parser = argparse.ArgumentParser(
            description="Print spread in angstrom.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--range",
                        nargs='+',
                        help="Range fo frames to analyse.",
                        type=int,
                        default=[0, 1, 1]
                        )
    args = parser.parse_args()

    _load = system.Supercell(**vars(args))
    for _fr in _load.XYZ:
        print(f'Frame {_fr}')
        _load.XYZ.get_atom_spread()


if __name__ == "__main__":
    main()
