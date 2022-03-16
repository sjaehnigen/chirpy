#!/usr/bin/env python
# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2022, The ChirPy Developers.
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
# -------------------------------------------------------------------


import argparse
import numpy as np
from matplotlib import pyplot as plt

from chirpy.classes import system
from chirpy.topology import grid


def main():
    '''Plots vibrational modes.'''
    parser = argparse.ArgumentParser(
        description="Plots vibrational modes as IR spectrum (optional: VCD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
            "fn",
            help="Input file containing vibrational modes."
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. orca, xvibs; optional).",
            default=None,
            )
    parser.add_argument(
            "--fwhm",
            help="Full width half maximum of the lineshape function (in 1/cm)",
            default=10.0,
            type=float,
            )
    parser.add_argument(
            "--lineshape",
            help="Lineshape function to use.",
            default='lorentzian'
            )
    parser.add_argument(
            "--save",
            help="Save results to file.",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--ignore_intensities",
            help="Set mode intensities to 1.",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--vcd",
            help="Plot VCD spectrum",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--xrange",
            nargs=2,
            help="Plotted range of frequencies (in 1/cm).",
            default=[4000., 0.],
            type=float,
            )
    parser.add_argument(
            "--resolution",
            help="Distance between data points (in 1/cm).",
            default=1.,
            type=float,
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='spectrum.dat'
            )
    args = parser.parse_args()

    i_fmt = args.input_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
        if i_fmt == 'hess':
            # --- assuming ORCA format
            i_fmt = 'orca'
    _load = system.Molecule(args.fn, fmt=i_fmt)

    _x0, _x1 = sorted(args.xrange)
    _n = int(abs(_x1 - _x0) / args.resolution)
    X = np.linspace(_x0, _x1, _n)

    if args.ignore_intensities:
        _load.Modes.IR_kmpmol = np.ones_like(_load.Modes.IR_kmpmol)

    data = {}
    data['va'] = (grid.regularisation(
                            _load.Modes.eival_cgs,
                            X,
                            args.fwhm,
                            mode=args.lineshape+"_std"
                            )
                  * _load.Modes.IR_kmpmol[:, None]
                  ).sum(axis=0)

    if args.vcd:
        data['vcd'] = (grid.regularisation(
                            _load.Modes.eival_cgs,
                            X,
                            args.fwhm,
                            mode=args.lineshape+"_std"
                            )
                       * _load.Modes.VCD_kmpmol[:, None]
                       ).sum(axis=0)

    # --- plot
    labels = {
       'va': ('Vibrational absorption spectrum', 'A in km/mol'),
       'vcd': ('Vibrational circular dichroism spectrum', r'$\Delta$A in ...'),
    }
    for _i in data:
        plt.plot(X, data[_i])
        plt.xlim(*args.xrange)
        plt.xlabel(r'$\tilde\nu$ in cm$^{-1}$')
        plt.title(labels[_i][0])
        plt.ylabel(labels[_i][1])
        plt.show()

        if args.save:
            np.savetxt(
                _i + '_' + args.f,
                np.array((X, data[_i])).T,
                header='omega in cm-1, %s' % _i
                )


if __name__ == "__main__":
    main()
