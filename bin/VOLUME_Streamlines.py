#!/usr/bin/env python

import argparse
import copy

from chirpy.classes import volume, trajectory
from chirpy.physics import constants


def main():
    parser = argparse.ArgumentParser(
            description="Trace vector field with seeded particles. Optional\
                        streamlines export to VMD.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "vector_field",
            help="File(s) containing velocity field (X,Y,Z) in atomic units.",
            nargs='+',
            )
    parser.add_argument(
            "--normalise",
            action='store_true',
            help="Normalise vector field with given scalar field",
            default=False
            )
    parser.add_argument(
            "--auto_crop",
            action='store_true',
            help="Reduce grid boundaries based on scalar threshold",
            default=False
            )
    parser.add_argument(
            "--crop_thresh",
            help="Absolute scalar threshold auto cropping grid.",
            type=float,
            default=5.E-3
            )
    parser.add_argument(
            "--scalar_field",
            help="File containing scalar field (optional)",
            default=None,
            )
    parser.add_argument(
            "--seed_points",
            help="File containing starting positions of streamlines particles.\
                  If None, the given scalar field is used.",
            default=None,
            )
    parser.add_argument(
            "--scalar_seed_sparse",
            help="If --seed_points is None.\
                  Sparsity of used grid points on scalar field.",
            type=int,
            default=3
            )
    parser.add_argument(
            "--scalar_seed_thresh",
            help="If --seed_points is None.\
                  Absolute threshold for selecting grid points on scalar\
                  field.",
            type=float,
            default=2.E-1
            )
    parser.add_argument(
            "--particles",
            help="File containing external particles' positions and velocities\
                  (optional)",
            default=None,
            )
    parser.add_argument(
            "--streamlines_sparse",
            help="Sparsity of velocity field grid points for propagation.",
            type=int,
            default=2
            )
    parser.add_argument(
            "--streamlines_length",
            help="Number of propagation steps.",
            type=int,
            default=20
            )
    parser.add_argument(
            "--streamlines_step",
            help="Propagation time step.",
            type=float,
            default=0.5
            )
    parser.add_argument(
            "--streamlines_direction",
            help="Choose 'forward', 'backward' propagation or 'both' (doubles\
                  length).",
            default='both'
            )
    args = parser.parse_args()

    # --- LOAD
    _vec = volume.VectorField(*args.vector_field)

    if args.scalar_field is not None:
        _sca = volume.ScalarField(args.scalar_field)

    # --- PRE-PROCESS
    if args.auto_crop:
        if args.scalar_field is None:
            raise AttributeError('Please specify --scalar_field for automatic'
                                 ' croppping!')
        _r = _sca.auto_crop(thresh=args.crop_thresh)  # Default: 1.E-3
        _vec.crop(_r)
    if args.normalise:
        if args.scalar_field is None:
            raise AttributeError('Please specify --scalar_field for'
                                 ' normalisation!')
        _vec.normalise(norm=_sca.data)

    _vec.print_info()

    # --- class operations
    # _vec.helmholtz_decomposition()

    # --- STREAMLINES AND ICONS

    if args.seed_points is None:
        if args.scalar_field is None:
            raise AttributeError('Please specify --scalar_field for automatic'
                                 ' points seed!')

        # --- Prepare initial points (pn) from scalar field cutoff + sparsity
        _sca.sparsity(args.scalar_seed_sparse)
        pn = (_sca.pos_grid()[:, _sca.data > args.scalar_seed_thresh]
              ).reshape((3, -1)).T
        print(f"Seeding {pn.shape[0]} points.")
        del _sca

    # --- streamline options
    export_args = {}
    if args.particles is not None:
        _part = trajectory.XYZFrame(args.particles)
        export_args.update({
                'external_object': True,
                'ext_p': copy.deepcopy(_part.pos_aa * constants.l_aa2au),
                'ext_v': copy.deepcopy(_part.vel_au),
                })
        del _part
    export_args.update({
                "sparse": args.streamlines_sparse,
                "forward": args.streamlines_direction in ["forward", "both"],
                "backward": args.streamlines_direction in ["backward", "both"],
                "length": args.streamlines_length,
                "timestep": args.streamlines_step
                })

    # --- generate streamlines
    _SL = _vec.streamlines(pn, **export_args)
    if args.particles is not None:
        traj, atoms = _SL
    else:
        traj = _SL

    # --- OUTPUT

    # --- convert positions to angstrom (velocities in a.u.!)
    traj[:, :, :3] *= constants.l_au2aa
    trajectory._XYZTrajectory(
                             data=traj,
                             symbols=traj.shape[1]*['C'],
                             comments=traj.shape[0]*['C']
                             ).write('streamlines.xyz')

    if args.particles is not None:
        atoms[:, :, :3] *= constants.l_au2aa
        trajectory._XYZTrajectory(
                             data=atoms,
                             symbols=_part.symbols,
                             comments=atoms.shape[0]*['C']
                             ).write('atoms.xyz')

    print('Done.')


if __name__ == "__main__":
    main()
