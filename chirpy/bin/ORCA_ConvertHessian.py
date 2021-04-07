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
#  Copyright (c) 2010-2021, The ChirPy Developers.
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
    '''Converts a Orca Hessian file into a XVIBS vibration file'''
    parser = argparse.ArgumentParser(
        description="Converts a Orca Hessian file into a XVIBS vibration file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
            "fn",
            help=".hess file from Orca output"
            )
    parser.add_argument(
            "-f",
            help="Output file name (the format is read from file extension "
                 "*.xvibs, *.molden)",
            default='output.xvibs'
            )
    args = parser.parse_args()

    _load = system.Molecule(args.fn, fmt='orca')
    _load.Modes.write(args.f)


if(__name__ == "__main__"):
    main()
