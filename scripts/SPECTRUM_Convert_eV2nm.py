#!/usr/bin/env python
# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------


import argparse
import numpy as np
from chirpy import constants


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
