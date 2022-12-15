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
from chirpy.classes import system
from chirpy import config


def main():
    '''Converts vibrational modes.'''
    parser = argparse.ArgumentParser(
        description="Converts vibrational modes.",
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
            "--output_format",
            help="Output file format (e.g. xvibs, molden, posvel (alias: xyz),\
                  traj; optional).",
            default=None,
            )
    parser.add_argument(
            "--modelist",
            nargs='+',
            help="List of modes (XYZ output only; 0-based, default: all).",
            type=int,
            default=None
            )
    parser.add_argument(
            "--factor",
            help="Velocity scaling, good for inversion (XYZ output only)",
            default=1.0,
            type=float,
            )
    parser.add_argument(
            "--n_images",
            help="Number (odd) of image frames to be calculated from vibration"
                 " (traj output only).",
            type=int,
            default=3)
    parser.add_argument(
            "--ts",
            help="Time step in fs (traj output only).",
            type=float,
            default=1,
            )
    parser.add_argument(
            "--T",
            help="Temperature for calculation of nuclear velocities "
                 "(xyz output only).",
            type=float,
            default=300,
            )
    parser.add_argument(
            "--mw",
            action="store_true",
            help="Assume modes as mass-weighted displacements."
                 "(xvibs input only; convention: False)",
            default=False
            )
    parser.add_argument(
            "--au",
            action="store_true",
            help="Assume atomic units in file."
                 "(xvibs input only; convention: False)",
            default=False
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='output'
            )
    parser.add_argument(
            "--n_runs",
            help="No. of runs in input file (for Gaussian only)",
            type=int,
            default=1
            )
    parser.add_argument(
            "--verbose",
            action='store_true',
            help="Print info and progress.",
            default=False,
            )
    args = parser.parse_args()
    config.set_verbose(args.verbose)

    i_fmt = args.input_format
    o_fmt = args.output_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
        if i_fmt == 'hess':
            # --- assuming ORCA format
            i_fmt = 'orca'

        if i_fmt == 'log':
            # --- assuming Gaussian format
            i_fmt = 'g09'
    _load = system.Molecule(args.fn, fmt=i_fmt, mw=args.mw, au=args.au,
                            run=args.n_runs)

    if o_fmt is None:
        o_fmt = args.f.split('.')[-1].lower()
    else:
        if o_fmt == 'posvel':
            o_fmt = 'xyz'
        if args.f == 'output':
            args.f += '.' + o_fmt

    if o_fmt not in ['cpmd', 'xyz', 'traj']:
        _load.Modes.write(args.f, fmt=o_fmt)

    else:
        _load.Modes.calculate_nuclear_velocities(temperature=args.T)
        if o_fmt in ['xyz', 'cpmd']:
            if args.modelist is not None:
                _load.Modes._modelist(args.modelist)
            _load.Modes.write(args.f, fmt=o_fmt, factor=args.factor)

        elif o_fmt == 'traj':
            if args.modelist is None:
                args.modelist = list(range(_load.Modes.n_modes))

            for _m in args.modelist:
                _fn_out = '%03d_' % _m + args.f
                _load.Modes.get_mode(_m).make_trajectory(
                                            n_images=args.n_images,
                                            ts_fs=args.ts,
                                            ).write(_fn_out,
                                                    fmt='xyz')


if(__name__ == "__main__"):
    main()
