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
from chirpy.interface import orca
from chirpy.write.modes import xvibsWriter
from chirpy.physics import constants


def main():
    '''Converts a Orca Hessian file into a XVIBS vibration file'''
    parser = argparse.ArgumentParser(
        description = "Converts a Orca Hessian file into a XVIBS vibration file",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fn", help=".hess file from Orca output")
    parser.add_argument("-f", help="Output file name", default='output.xvibs')
    args = parser.parse_args()

    if not os.path.exists(args.fn):
        raise FileNotFoundError('File %s does not exist' % args.fn)

    symbols, pos_au, freqs, modes, modes_res = orca.read_hessian_file(args.fn)

    numbers = constants.symbols_to_numbers(symbols)
    xvibsWriter(args.f, len(symbols), numbers, pos_au*constants.l_au2aa, freqs, modes_res)

if(__name__ == "__main__"):
    main()

#EOF
