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
