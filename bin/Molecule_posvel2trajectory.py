#!/usr/bin/env python

import argparse
import numpy as np
from chemsnail.classes import system
from chemsnail.topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported vib input into normal mode traj", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="xyz")
    parser.add_argument("-f", help="Output file name.", default='traj.xyz')
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-cell_aa_deg", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-n_images", help="Number of image frames to be calculated from vibration (use odd number).", default=3)
    parser.add_argument("-ts", help="Time step in fs.", default=1)
    args = parser.parse_args()
    
    if int(args.n_images)%2 == 0:
        raise AttributeError('Number of images has to be an odd number!')
    args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    _mol = system.Molecule(**vars(args)
                          ).XYZData._make_trajectory(n_images=int(args.n_images),
                                                     ts_fs=float(args.ts)
                                                    )
    _mol._wrap_atoms(args.cell_aa_deg)
#    _mol._wrap_molecules([0]*12,args.cell_aa_deg)                                                             
    _mol.write(args.f)

if(__name__ == "__main__"):
    main()
