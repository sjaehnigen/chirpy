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
from chirpy.classes import system


def main():
    '''Create a topology file from input.'''
    parser = argparse.ArgumentParser(
        description="Create a topology file from input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
            "fn",
            help="Input structure file (xyz.pdb,xvibs,...)"
            )
    parser.add_argument(
        "--center_coords",
        action='store_true',
        help="Center coordinates in cell centre or at origin "
             "(default: false; box_aa parametre overrides default origin).",
        default=False,
        )
    parser.add_argument(
        "--cell_aa_deg",
        nargs=6,
        help="Orthorhombic cell parametres a b c al be ga in angstrom/degree.",
        type=float,
        default=None
        )
    parser.add_argument(
        "--wrap_mols",
        action='store_true',
        help="Wrap molecules instead of atoms in cell.",
        default=False
        )
    parser.add_argument("-f", help="Output file name", default='out.pdb')
    args = parser.parse_args()

    _load = system.Molecule(**vars(args))
    _load.install_molecular_origin_gauge()

    if args.wrap_mols:
        _load.wrap_molecules()
    else:
        _load.wrap_atoms()
    _load.write(args.f, fmt='pdb')


if(__name__ == "__main__"):
    main()
