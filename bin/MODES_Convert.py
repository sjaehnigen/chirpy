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
from chirpy.classes import system


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
            help="Output file format (e.g. xvibs, molden, posvel; optional).",
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
            help="Temperature for calculation of nuclear velocities (xyz output only).",
            type=float,
            default=300,
            )
    parser.add_argument(
            "--mw",
            action="store_true",
            help="Assume modes as mass-weighted displacements "
                 "(xvibs input only; convention: False)",
            default=False
            )
    parser.add_argument(
            "-f",
            help="Output file name (the format is read from file extension "
                 "*.xvibs, *.molden)",
            default='output.xvibs'
            )
    args = parser.parse_args()

    i_fmt = args.input_format
    o_fmt = args.output_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
        if i_fmt == 'hess':
            # --- assuming ORCA format
            i_fmt = 'orca'
    _load = system.Molecule(args.fn, fmt=i_fmt)

    if o_fmt is None:
        o_fmt = args.f.split('.')[-1].lower()

    if o_fmt not in ['xyz', 'posvel', 'traj']:
        _load.Modes.write(args.f, fmt=o_fmt)

    else:
        _load.Modes.calculate_nuclear_velocities(temperature=args.T)

        if o_fmt in ['xyz', 'posvel']:
            if args.modelist is not None:
                _load.Modes._modelist(args.modelist)
            _load.Modes.write(args.f, fmt='xyz', factor=args.factor)

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
