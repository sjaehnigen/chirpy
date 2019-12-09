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
    '''Align a line that connects i0 and i1 to an axis.'''
    parser = argparse.ArgumentParser(
            description="Align a line that connects i0 and i1 to an axis.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--axis",
                        help="Vector to align to",
                        nargs=3,
                        type=float,
                        default=[0., 0., 1.]
                        )
    parser.add_argument("-i0",
                        help="Origin of reference in system",
                        default=0,
                        type=int,
                        )
    parser.add_argument("-i1",
                        help="Tip of reference in system",
                        default=1,
                        type=int,
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )

    args = parser.parse_args()
    args.axis = np.array(args.axis)

    _system = system.Supercell(**vars(args)).XYZ
    _system.align_to_vector(args.i0, args.i1, args.axis)
    _system.write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
