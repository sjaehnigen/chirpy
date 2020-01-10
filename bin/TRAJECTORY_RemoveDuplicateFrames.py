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
    '''Scan for duplicate frames and write new trajectory. BETA'''
    parser = argparse.ArgumentParser(
            description="Scan for duplicate frames and write new trajectory.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )

    args = parser.parse_args()

    _system = system.Supercell(**vars(args)).XYZ
    _system.mask_duplicate_frames(verbose=True)
    _system.write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
