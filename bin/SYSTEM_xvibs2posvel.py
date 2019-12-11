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
from chirpy.classes import system


def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser = argparse.ArgumentParser(
            description="Convert any supported vib input into cpmd trajectory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="file (xvibs,...)"
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='posvel.xyz'
            )
    parser.add_argument(
            "-mw",
            action='store_true',
            help="Assume mass-weighed coordinates in fn",
            default=False
            )
    parser.add_argument(
            "--format",
            help="File format (default: from file extension)",
            default=None
            )
    parser.add_argument(
            "--modelist",
            nargs='+',
            help="List of modes  (0-based index, default: all).",
            default=[None]
            )
    parser.add_argument(
            "--factor",
            help="Velocity scaling, good for inversion",
            default=1.0
            )

    args = parser.parse_args()
    args.factor = float(args.factor)
    if args.format is not None:
        args.fmt = args.format

    if not any(args.modelist):
        system.Molecule(**vars(args)
                        ).Modes.write_nuclear_velocities(args.f,
                                                         fmt='xyz',
                                                         factor=args.factor)
    else:
        system.Molecule(**vars(args)
                        ).Modes.write_nuclear_velocities(args.f,
                                                         fmt='xyz',
                                                         modelist=[int(m) for m in args.modelist],
                                                         factor=args.factor)


if(__name__ == "__main__"):
    main()
