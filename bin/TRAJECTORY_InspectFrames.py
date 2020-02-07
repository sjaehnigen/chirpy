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
import warnings

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
    parser.add_argument("--rewrite",
                        help="Write new file with cleaned data",
                        action='store_true',
                        default=False,
                        )

    args = parser.parse_args()

    _system = system.Supercell(**vars(args))
    _system.print_info()
    _system = _system.XYZ
    print('Checking for duplicate frames...')
    _system.mask_duplicate_frames(verbose=True, rewind=False)
    print('Done.\n')

    print('Total no. frames (inluding duplicates):', _system._fr)
    if args.rewrite:
        warnings.warn('Please use TRAJECTORY_Convert.py for rewriting data!',
                      stacklevel=2)


if __name__ == "__main__":
    main()
