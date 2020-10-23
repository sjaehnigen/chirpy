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


import argparse
import numpy as np

from chirpy.classes import system


def create_name(F):
    return F.split('.')[0] + '_last_step' + '.xyz'


def main():
    '''Extract last frame from a trajectory.'''
    parser = argparse.ArgumentParser(
            description="Extract last frame from a trajectory.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--cell_aa_deg",
                        nargs=6,
                        help="Orthorhombic cell parametres a b c al be ga in \
                                angstrom/degree",
                        default=[0.0, 0.0, 0.0, 90., 90., 90.]
                        )
    parser.add_argument("-f",
                        help="Output XYZ file name (auto: 'fn_[step].xyz')",
                        default=create_name
                        )

    args = parser.parse_args()
    args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)

    if args.f is create_name:
        args.f = create_name(args.fn)

    _load = system.Supercell(**vars(args)).XYZ
    _load._unwind()
    _load._frame.write(args.f)


if __name__ == "__main__":
    main()
