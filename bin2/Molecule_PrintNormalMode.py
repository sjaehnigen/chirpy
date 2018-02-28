#!/usr/bin/python3

import argparse
import numpy as np 
from classes import molecule
from topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported vib input into normal mode traj", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (only xvibs supported)")
    parser.add_argument("-f", help="Output file name (standard: 'mode.xyz')", default='mode.xyz')
    parser.add_argument("-mw", action='store_true', help="Assume mass-weighed coordinates in fn (default: false).", default=False)
    parser.add_argument("-ignore_warnings", action='store_true', help="Ignore warnings regarding orthonormality (default: false).", default=False)
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-modelist", nargs='+', help="List of modes  (0-based index, default: all).", default=[None])
    parser.add_argument("-cell_aa", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-n_images", help="Number of image frames to be calculated from vibration (default: 3, use odd number).", default=3)
    parser.add_argument("-ts", help="Time step in fs (default: 1).", default=1)

    args = parser.parse_args()
    args.cell_aa = np.array(args.cell_aa).astype(float)
    if not any(args.modelist):
        molecule.Molecule(fmt='xvibs',
                          **vars(args)
                          ).Modes.print_modes(args.f,fmt='xyz',n_images=int(args.n_images),ts_fs=float(args.ts))
    else:
        molecule.Molecule(fmt='xvibs',
                          **vars(args)
                          ).Modes.print_modes(args.f,fmt='xyz',modelist=[int(m) for m in args.modelist],n_images=int(args.n_images),ts_fs=float(args.ts))


if(__name__ == "__main__"):
    main()
