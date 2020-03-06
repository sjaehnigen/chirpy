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
            nargs='+',
            help="Center atom list (id starting from 0) in cell \
                    and wrap or \'True\' for selecting all atoms.",
            default=None,
            )
    parser.add_argument(
            "--use_com",
            action='store_true',
            help="Use centre of mass instead of centre of geometry \
                    as reference",
            default=False
            )
    parser.add_argument(
        "--cell_aa_deg",
        nargs=6,
        help="Orthorhombic cell parametres a b c al be ga in angstrom/degree.",
        type=float,
        default=None
        )
    parser.add_argument(
        "--wrap_molecules",
        action='store_true',
        help="Wrap molecules instead of atoms in cell.",
        default=False
        )
    parser.add_argument("-f", help="Output file name", default='out.pdb')
    args = parser.parse_args()

    _load = system.Molecule(**vars(args))
    _load.define_molecules()

    if args.wrap_molecules:
        _load.wrap_molecules()
    else:
        _load.wrap_atoms()
    _load.write(args.f)


if(__name__ == "__main__"):
    main()
