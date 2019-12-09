#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


import argparse
import numpy as np
from chirpy.create.supercell import MolecularCrystal


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
        nargs['install_mol_gauge'] = args.get_mols

    b = MolecularCrystal.read(args.fn, **nargs)
    b.print_info()

    c = MolecularCrystal()
    c.print_info()

    b.write(multiply=(1, 2, 1))
    

if __name__ == "__main__":
    main()
