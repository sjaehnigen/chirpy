#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


import argparse
import numpy as np
from chirpy.classes import system

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Calculate velocities from a trajectory (beta)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xyz.pdb,xvibs,...)")
    parser.add_argument("--separate_files", action='store_true', help="Write one file for each frame (default: false).", default=False)
    parser.add_argument("--align_atoms", action='store_true', help="Align atoms with reference to first frame (default: false).", default=False)
    parser.add_argument("--center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("--cell_aa", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-f", help="Output file name (standard: 'TRAJSAVED')", default='TRAJSAVED')
    parser.add_argument("-pp", help="Pseudopotential (default: 'MT_BLYP')", default='MT_BLYP')
    parser.add_argument("-bs", help="Method for treatment of the non-local part of the pp (default: '')", default='')
    parser.add_argument("--factor", type=float, help="Velocity scaling, good for inversion (default: 1.0).", default=1.0)
    parser.add_argument("-ts", type=float, help="Time step between frames in fn.", default=0.5)
    args = parser.parse_args()
    args.cell_aa = np.array(args.cell_aa).astype(float)
    sys = system.Molecule(**vars(args))
    sys.XYZ.calculate_nuclear_velocities(**vars(args)) # this actions could be send as var to class
    #sys.XYZ.write(args.f,fmt='cpmd',pp=args.pp,bs=args.bs,factor=args.factor)
    sys.XYZ.write(args.f,fmt='xyz',attr='data',factor=args.factor,separate_files=args.separate_files)


if(__name__ == "__main__"):
    main()
