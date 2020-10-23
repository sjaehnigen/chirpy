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
from chirpy.create import supercell


def main():
    parser = argparse.ArgumentParser(
            description="Read supercell and print box information.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="supercell (xyz, pdb, xvibs, ...)"
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Cell parametres a b c al be ga in angstrom/degree \
(None = guess from atom spread).",
            default=None,
            type=float
            )
    parser.add_argument(
            "--fn_topo",
            help="Topology pdb file (optional)",
            default=None
            )
    parser.add_argument(
            "--get_mols",
            action="store_true",
            help="Find molecules in cell (slow for medium/large systems)"
            )
    args = parser.parse_args()

    nargs = {}
    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        nargs['cell_aa_deg'] = np.array(args.cell_aa_deg).astype(float)

    nargs['fn_topo'] = args.fn_topo
    if args.get_mols:
        nargs['define_molecules'] = args.get_mols

    # --- ToDo: define type in arguments
    b = supercell.Solution.read(args.fn, **nargs)
    b.print_info()


if __name__ == "__main__":
    main()
