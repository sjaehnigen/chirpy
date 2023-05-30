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


def main():
    '''Evaluate Sum Rules for APT and AAT'''
    parser = argparse.ArgumentParser(
            description="Evaluate Sum Rules for APT and AAT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "sigma_0",
            help=""
            )
    parser.add_argument(
            "sigma_dip",
            help=""
            )
    parser.add_argument(
            "sigma_apt",
            help=""
            )
    parser.add_argument(
            "sigma_aat",
            help=""
            )
    # parser.add_argument(
    #         "--verbose",
    #         action='store_true',
    #         help="Print info and progress.",
    #         default=False,
    #         )
    args = parser.parse_args()
    # config.set_verbose(args.verbose)

    # if args.cell_aa_deg is None:
    #     del args.cell_aa_deg

    sigma_0 = np.loadtxt(args.sigma_0).astype(float).reshape((3, 3))
    sigma_dip = np.loadtxt(args.sigma_dip).astype(float).reshape((3, 3))
    sigma_apt = np.loadtxt(args.sigma_apt).astype(float).reshape((3, 3))
    sigma_aat = np.loadtxt(args.sigma_aat).astype(float).reshape((3, 3))

    print('sigma 0')
    print(sigma_0)
    print('')
    print('sigma 1')
    print(sigma_dip)
    print('')
    print('sigma 2')
    print(sigma_apt)
    print('')
    print('sigma 3')
    print(sigma_aat)
    print('')
    print(f'0       = {np.linalg.norm(sigma_0, "fro"):.5f}')
    print(f'bar 1   = {0.5*np.linalg.norm(sigma_dip**2+sigma_dip.T**2, "fro"):.5f}')
    print(f'delta 1 = {0.5*np.linalg.norm(sigma_dip**2-sigma_dip.T**2, "fro"):.5f}')
    print(f'bar 2   = {0.5*np.linalg.norm(sigma_apt**2+sigma_apt.T**2, "fro"):.5f}')
    print(f'delta 2 = {0.5*np.linalg.norm(sigma_apt**2-sigma_apt.T**2, "fro"):.5f}')
    print(f'bar 3   = {0.5*np.linalg.norm(sigma_aat**2+sigma_aat.T**2, "fro"):.5f}')
    print(f'delta 3 = {0.5*np.linalg.norm(sigma_aat**2-sigma_aat.T**2, "fro"):.5f}')


if __name__ == "__main__":
    main()
