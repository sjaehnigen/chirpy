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
            "--decompose",
            action='store_true',
            help="Calculate and return the Helmholtz decomposed vector fields\
                  as well (before normalisation)",
            default=False
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
            "--sparse",
            help="Sparsity of velocity field grid points for propagation.",
            type=int,
            default=1
            )
    parser.add_argument(
            "--length",
            help="Number of propagation steps.",
            type=int,
            default=20
            )
    parser.add_argument(
            "--step",
            help="Propagation time step in fs.",
            type=float,
            default=1.0
            )
    parser.add_argument(
            "--direction",
            help="Choose 'forward', 'backward' propagation or 'both' (doubles\
                  length).",
            default='both'
            )
    parser.add_argument(
            "-f",
            help="Output file stem for output",
            default='streamlines'
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
        _r = _sca.auto_crop(thresh=args.crop_thresh)
        _vec.crop(_r)

    if args.decompose:
        _vec.helmholtz_decomposition()

    if args.normalise:
        if args.scalar_field is None:
            raise AttributeError('Please specify --scalar_field for'
                                 ' normalisation!')
        _vec.normalise(norm=_sca)
        if args.decompose:
            _vec.irrotational_field.normalise(norm=_sca)
            _vec.solenoidal_field.normalise(norm=_sca)
            _vec.homogeneous_field.normalise(norm=_sca)

    _vec.print_info()

    # --- STREAMLINES AND ICONS

    if args.seed_points is None:
        if args.scalar_field is None:
            raise AttributeError('Please specify --scalar_field for automatic'
                                 ' points seed!')

        # --- Prepare initial points (pn) from scalar field cutoff + sparsity
        _sca = _sca.sparse(args.scalar_seed_sparse)
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
                'ext_pos_aa': copy.deepcopy(_part.pos_aa),
                'ext_vel_au': copy.deepcopy(_part.vel_au),
                })
    export_args.update({
                "sparse": args.sparse,
                "forward": args.direction in ["forward", "both"],
                "backward": args.direction in ["backward", "both"],
                "length": args.length,
                "timestep_fs": args.step
                })

    # --- generate streamlines
    _SL = _vec.streamlines(pn, **export_args)
    traj = _SL['streamlines']
    if args.particles is not None:
        atoms = _SL['particles']

    if args.decompose:
        traj_div = _vec.irrotational_field.streamlines(
                                            pn, **export_args)['streamlines']
        traj_rot = _vec.solenoidal_field.streamlines(
                                            pn, **export_args)['streamlines']
        traj_hom = _vec.homogeneous_field.streamlines(
                                            pn, **export_args)['streamlines']

    # --- OUTPUT
    _stem = args.f

    # --- convert positions to angstrom (velocities in a.u.!)
    trajectory._XYZTrajectory(
                             data=traj,
                             symbols=traj.shape[1]*['C'],
                             comments=traj.shape[0]*['C']
                             ).write(_stem + '.xyz')
    if args.particles is not None:
        trajectory._XYZTrajectory(
                             data=atoms,
                             symbols=_part.symbols,
                             comments=atoms.shape[0]*['C']
                             ).write(_stem + '_atoms.xyz')

    if args.decompose:
        traj_div[:, :, :3] *= constants.l_au2aa
        traj_rot[:, :, :3] *= constants.l_au2aa
        traj_hom[:, :, :3] *= constants.l_au2aa
        trajectory._XYZTrajectory(
                             data=traj_div,
                             symbols=traj_div.shape[1]*['C'],
                             comments=traj_div.shape[0]*['C']
                             ).write(_stem + '_div.xyz')
        trajectory._XYZTrajectory(
                             data=traj_rot,
                             symbols=traj_rot.shape[1]*['C'],
                             comments=traj_rot.shape[0]*['C']
                             ).write(_stem + '_rot.xyz')
        trajectory._XYZTrajectory(
                             data=traj_hom,
                             symbols=traj_hom.shape[1]*['C'],
                             comments=traj_hom.shape[0]*['C']
                             ).write(_stem + '_hom.xyz')

    print('Done.')


if __name__ == "__main__":
    main()
