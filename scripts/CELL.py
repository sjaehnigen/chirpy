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
import numpy as np
import chirpy as cp

np.set_printoptions(precision=5, suppress=True)


def main():
    parser = argparse.ArgumentParser(
            description="Convert cell parameters.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "cell",
            nargs='+',
            help="Cell parameters a b c al be ga in angstrom/degree OR "
                 "cell vectors xa, ya, za, xb, yb, zb, xc, yc, zc angstrom.",
            default=None,
            type=float
            )
    args = np.array(parser.parse_args().cell)

    if len(args) == 6:
        cell = cp.topology.mapping.get_cell_vec(args)
    elif len(args) == 9:
        cell = cp.topology.mapping.get_cell_l_deg(args.reshape((3, 3)))
    print(cell)


if __name__ == "__main__":
    main()
