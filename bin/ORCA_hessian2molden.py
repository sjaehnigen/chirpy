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
import os
from chirpy.interface import orca, molden


def main():
    '''Converts a Orca Hessian file into a molden vibration file'''
    parser = argparse.ArgumentParser(
        description = "Converts a Orca Hessian file into a molden vibration file",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fn", help=".hess file from Orca output")
    parser.add_argument("-f", help="Output file name", default='output.mol')
    args = parser.parse_args()

    if not os.path.exists(args.fn):
        raise FileNotFoundError('File %s does not exist' % args.fn)

    symbols, pos_au, freqs, modes, modes_res = orca.read_hessian_file(args.fn)
    molden.WriteMoldenVibFile(args.f, symbols, pos_au, freqs, modes)

if(__name__ == "__main__"):
    main()

#EOF
