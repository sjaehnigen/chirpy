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
import numpy as np
from chirpy.physics import constants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fn_in',          default=None)
    parser.add_argument('-o',  '--fn_out',         default='out.dat')
    parser.add_argument('--no_header', action='store_true', default=False)
    args = parser.parse_args()

    data = np.loadtxt(args.fn_in)[1-int(args.no_header):].astype(float)
    print(data.shape)

    with open(args.fn_out, 'w') as f:
        f.write(''.join(["%12.6f %12.6f\n" % p
                         for p in zip(
                                data[:, 0] / constants.E_eV2cm_1,
                                data[:, 1]
                                )
                         ]))


if __name__ == "__main__":
    main()
