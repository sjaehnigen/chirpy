#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
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
                                map(lambda x: constants.E_J2nm(x),
                                    constants.E_eV2J * data[:, 0]),
                                data[:, 1]
                                )
                         ]))


if __name__ == "__main__":
    main()
