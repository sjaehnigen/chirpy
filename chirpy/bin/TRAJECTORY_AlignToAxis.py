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
import numpy as np
from chirpy.classes import system


def main():
    '''Align a line that connects i0 and i1 to an axis.'''
    parser = argparse.ArgumentParser(
            description="Align a line that connects atoms i0 and i1 to an "
                        "axis (no PBC support).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--axis",
                        help="Vector to align to",
                        nargs=3,
                        type=float,
                        default=[0., 0., 1.]
                        )
    parser.add_argument("-i0",
                        help="Atom index of reference origin in system",
                        default=0,
                        type=int,
                        )
    parser.add_argument("-i1",
                        help="Atom index of reference tip in system",
                        default=1,
                        type=int,
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )

    args = parser.parse_args()
    args.axis = np.array(args.axis)

    _system = system.Supercell(**vars(args)).XYZ
    _system.align_to_vector(args.i0, args.i1, args.axis)
    _system.write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
