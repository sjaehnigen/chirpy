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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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

from chirpy.classes import trajectory, volume
from chirpy.interface import vmd
from chirpy.physics import constants


def main():
    parser = argparse.ArgumentParser(
            description="Create VMD arrows from input. For advanced options,\
                         please use the VMDPaths object itnterface directly.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input file (trajectory, modes etc.).",
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. xyz, pdb, cpmd; optional).",
            default=None,
            )
    parser.add_argument(
            "--property",
            help="Choose content of FN to be visualised: "
                 "positions (trajectory/path), "
                 "mode (specified through --mode), "
                 "velocities (frame)"
                 "moments_c (frame)"
                 "moments_m (frame)"
                 "vectorfield",
            default='positions'
            )
    parser.add_argument(
            "--mode",
            help="Choosen mode (see --property).",
            type=int,
            default=0
            )
    parser.add_argument(
            "--scale",
            help="Scale vectors (velocities, moments, mode, ...; "
                 "see --property). For velocities this keyword corresponds to "
                 "the time step in fs",
            type=float,
            default=1.0
            )
    parser.add_argument(
            "--style",
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
            default="AOShiny"
            )
    parser.add_argument(
            "--frame",
            help="For vectors: select frame.",
            type=int,
            default=0
            )
    parser.add_argument(
            "--skip",
            help="For trajectories: use every <skip>th timestep.",
            type=int,
            default=1
            )
    parser.add_argument(
            "--sparse",
            help="For vectorfield: use every <sparse>th grid point.",
            type=int,
            default=2
            )
    parser.add_argument(
            "--cutoff_aa",
            help="Minimum length of arrow for being drawn in angstrom",
            type=float,
            default=0.1
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
    i_fmt = args.input_format
    largs = {}
    largs['range'] = (0, args.skip, 1e99)
    largs['frame'] = args.frame
    if i_fmt is not None:
        largs.update({'fmt': i_fmt})

    if args.property == 'positions':
        traj = trajectory._XYZTrajectory(args.fn, **largs)
        paths = vmd.VMDPaths(traj.pos_aa, auto_smooth=not args.no_smooth)

    elif args.property == 'velocities':
        traj = trajectory.XYZFrame(args.fn, **largs)
        paths = vmd.VMDPaths.from_vector(traj.pos_aa,
                                         traj.vel_au*constants.v_au2aaperfs,
                                         scale=args.scale)

    elif args.property == 'mode':
        traj = trajectory.VibrationalModes(args.fn, **largs)
        # --- ToDO: allow combination of modes with modelist
        paths = vmd.VMDPaths.from_vector(traj.pos_aa[args.mode],
                                         traj.modes[args.mode],
                                         scale=args.scale)

    elif args.property == 'moments_c':
        traj = trajectory.MOMENTSFrame(args.fn, **largs)
        paths = vmd.VMDPaths.from_vector(traj.pos_aa,
                                         traj.c_au*constants.v_au2aaperfs,
                                         scale=args.scale)

    elif args.property == 'moments_m':
        traj = trajectory.MOMENTSFrame(args.fn, **largs)
        paths = vmd.VMDPaths.from_vector(traj.pos_aa,
                                         traj.m_au*constants.v_au2aaperfs,
                                         scale=args.scale)

    elif args.property == 'vectorfield':
        traj = volume.VectorField(args.fn, **largs)
        # --- scale for max-norm
        args.scale *= 0.01
        args.cutoff_aa *= 0.01
        paths = vmd.VMDPaths.from_vector_field(traj,
                                               sparse=args.sparse,
                                               scale=args.scale,
                                               normalise='max')
    else:
        raise ValueError(f'unknown property {args.property}')

    # --- ARROWS for VMD
    # N.B.: It is WAY faster to source the tcl file from the vmd command line
    # ("source file.tcl") than loading it at startup ("-e file.tcl")!

    # --- OUTPUT
    if args.f is None:
        _stem = ''
    else:
        _stem = args.f + '_'

    nargs = {}
    if args.rgb is not None:
        nargs['rgb'] = tuple(args.rgb)

    if args.style == 'lines':
        paths.draw_line(_stem + 'lines.tcl',
                        sparse=1,  # args.skip,
                        arrow=not args.no_head,
                        cutoff_aa=args.cutoff_aa,
                        arrow_resolution=args.resolution,
                        # arrow_radius=args.radius,
                        width=args.linewidth,
                        **nargs
                        )

    if args.style == 'icons':
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
