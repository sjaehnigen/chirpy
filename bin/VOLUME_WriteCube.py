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
from chirpy.classes import volume

def main():
    '''Write volume data into file'''
    parser=argparse.ArgumentParser(
            description="Write volume data into file",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="File that contains volume data")
    parser.add_argument(
            "--sparsity",
            help="Reduce the meshsize by factor.",
            type=int,
            default=1
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='out.cube')
    args = parser.parse_args()

    fn = args.fn
    system = volume.ScalarField(fn, **vars(args))
    system.write(args.f)

if(__name__ == "__main__"):
    main()
