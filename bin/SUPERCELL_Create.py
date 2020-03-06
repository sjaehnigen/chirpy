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


import argparse
import numpy as np
from chirpy.create.supercell import MolecularCrystal


def main():
    parser = argparse.ArgumentParser(
            description="Read coordinates and create a new supercell (WORK IN PROGESS).",
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
            "--multiply",
            nargs=3,
            help="Multilee unit cells in X, Y, Z.",
            default=[1, 1, 1],
            type=int
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
    parser.add_argument(
            "--wrap_molecules",
            action='store_true',
            help="Wrap molecules instead of atoms in cell.",
            default=False
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='supercell.pdb'
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

    b = MolecularCrystal.read(args.fn, **nargs)
    b.print_info()

    b.write(args.f,
            multiply=tuple(args.multiply),
            wrap_molecules=args.wrap_molecules
            )


if __name__ == "__main__":
    main()
