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
    '''Convert any supported input into XYZ format'''
    parser = argparse.ArgumentParser(
            description="Convert any supported input into XYZ format",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--center_coords",
                        nargs='+',
                        help="Center atom list (id starting from 0) in cell \
                                and wrap",
                        default=None,
                        type=int,
                        )
    parser.add_argument("--align_coords",
                        nargs='+',
                        help="Align atom list (id starting from 0)",
                        default=None,
                        type=int,
                        )
    parser.add_argument("--use_com",
                        action='store_true',
                        help="Use centre of mass instead of centre of geometry \
                                as reference",
                        default=False
                        )
    parser.add_argument("--wrap",
                        action='store_true',
                        help="Wrap atoms in cell.",
                        default=False
                        )
    parser.add_argument("--cell_aa_deg",
                        nargs=6,
                        help="Use custom cell parametres a b c al be ga in \
                                angstrom/degree",
                        default=None,
                        )
    parser.add_argument("--range",
                        nargs=3,
                        help="Range of frames to read (start, step, stop)",
                        default=None,
                        type=int,
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )
    args = parser.parse_args()

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    if args.range is None:
        args.range = (0, 1, float('inf'))

    system.Supercell(**vars(args)).write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
