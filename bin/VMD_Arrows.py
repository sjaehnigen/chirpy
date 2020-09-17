#!/usr/bin/env python

import argparse

from chirpy.classes import trajectory
from chirpy.interface import vmd


def main():
    parser = argparse.ArgumentParser(
            description="Create VMD arrows from input. For advanced options,\
                         please use the VMDPaths object itnterface directly.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input file (trajectory, modes, velocities, etc.).",
            )
    parser.add_argument(
            "--type",
            help="Choose 'lines' or 'icons'",
            default='lines'
            )
    parser.add_argument(
            "--no_head",
            action='store_true',
            help="Do not draw arrow heads",
            default=False
            )
    parser.add_argument(
            "--linewidth", "-lw",
            help="For lines only.",
            type=int,
            default=1
            )
    parser.add_argument(
            "--radius",
            help="Radius of icons (lines: radius of arrow head).",
            type=float,
            default=0.008
            )
    parser.add_argument(
            "--resolution",
            help="Resolution of icons (lines: resolution of arrow head).",
            type=int,
            default=10
            )
    parser.add_argument(
            "--rgb",
            help="RGB code of line/icon colour. Format: r g b",
            nargs=3,
            type=float,
            default=None
            )
    parser.add_argument(
            "--material",
            help="Draw material for icons",
            default="Diffuse"
            )
    parser.add_argument(
            "--skip",
            help="For trajectories: use every <skip>th timestep.",
            type=int,
            default=1
            )
    parser.add_argument(
            "--cutoff_aa",
            help="Minimum length of arrow for being drawn in angstrom",
            type=float,
            default=0.5
            )
    parser.add_argument(
            "--no_smooth",
            action='store_true',
            help="Skip spline smoothing of path",
            default=False
            )
    parser.add_argument(
            "-f",
            help="Output file stem for output",
            default=None
            )

    args = parser.parse_args()

    traj = trajectory._XYZTrajectory(args.fn, range=(0, args.skip, 1e99))

    # ToDo: Modes and velocities
    #          for _m in args.modelist:
    #              _fn_out = '%03d_' % _m + args.f
    #              _load.Modes.get_mode(_m).make_trajectory(
    #                                          n_images=args.n_images,
    #                                          ts_fs=args.ts,
    #                                          ).write(_fn_out,
    #                                                  fmt='xyz')

    # --- ARROWS for VMD
    # N.B.: It is WAY faster to source the tcl file from the vmd command line
    # ("source file.tcl") than loading it at startup ("-e file.tcl")!

    paths = vmd.VMDPaths(traj.pos_aa, auto_smooth=not args.no_smooth)

    # --- OUTPUT
    if args.f is None:
        _stem = ''
    else:
        _stem = args.f + '_'

    nargs = {}
    if args.rgb is not None:
        nargs['rgb'] = tuple(args.rgb)

    if args.type == 'lines':
        paths.draw_line(_stem + 'lines.tcl',
                        sparse=1,  # args.skip,
                        arrow=not args.no_head,
                        cutoff_aa=args.cutoff_aa,
                        arrow_resolution=args.resolution,
                        arrow_radius=args.radius,
                        **nargs
                        )

    if args.type == 'icons':
        nargs['material'] = args.material
        paths.draw_tube(_stem + 'icons.tcl',
                        sparse=1,  # args.skip,
                        arrow=not args.no_head,
                        cutoff_aa=args.cutoff_aa,
                        resolution=args.resolution,
                        radius=args.radius,
                        **nargs
                        )


if __name__ == "__main__":
    main()
