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
from chirpy.classes import volume


def main():
    '''Converts CUBE kernel into numpy array on disk \
            (beta version; no cell parametres conserved!)'''
    parser = argparse.ArgumentParser(
        description="Converts CUBE kernel into numpy array on disk \
                (beta version; no cell parametres conserved!)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
            "fn",
            help="cube file"
            )
    parser.add_argument(
            "--crop",
            help="Crop grid with integer",
            default=None
            )
    parser.add_argument(
            "-f",
            help="Output file name (default: <cube-file-stem>.npy)",
            default=None
            )
    args = parser.parse_args()

    _S = volume.ScalarField(args.fn)

    if args.crop is not None:
        _S.crop(args.crop)

    if args.f is None:
        _out = ''.join(args.fn.split('.')[:-1])
    else:
        _out = args.f

    np.save(_out, _S.data)


if(__name__ == "__main__"):
    main()

# EOF
