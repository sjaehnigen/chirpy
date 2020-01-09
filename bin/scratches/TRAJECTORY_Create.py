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
import numpy as np
from chirpy.classes import system


# clean up, add fn_vel, range etc...

def main():
    '''Convert any supported vib input into normal mode traj.'''
    parser = argparse.ArgumentParser(
          description="Linearly propagate input date to create a trajectory.",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter
          )
    parser.add_argument("fn", help="xyz")
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).",
            default=[0.0, 0.0, 0.0, 90., 90., 90.]
            )
    parser.add_argument(
            "--n_images",
            help="Number of image frames to be calculated from vibration (use odd number).",
            default=3,
            type=int
            )
    parser.add_argument(
            "-ts",
            help="Time step in fs.",
            type=float,
            default=1
            )
    parser.add_argument(
            "-f",
            help="Output file name.",
            default='traj.xyz'
            )
    args = parser.parse_args()

    if int(args.n_images) % 2 == 0:
        raise ValueError('Number of images has to be an odd number!')
    args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    _mol = system.Molecule(**vars(args)
                           ).XYZ.make_trajectory(n_images=int(args.n_images),
                                                 ts_fs=float(args.ts)
                                                 )
    _mol.wrap_atoms(args.cell_aa_deg)
    _mol.write(args.f)


if(__name__ == "__main__"):
    main()
