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
    '''Print spread in angstrom.'''
    parser = argparse.ArgumentParser(
            description="Print spread in angstrom.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--range",
                        nargs='+',
                        help="Range fo frames to analyse.",
                        type=int,
                        default=[0, 1, 1]
                        )
    args = parser.parse_args()

    _load = system.Supercell(**vars(args))
    for _fr in _load.XYZ:
        print(f'Frame {_fr}')
        _load.XYZ.get_atom_spread()


if __name__ == "__main__":
    main()
