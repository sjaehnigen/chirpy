#!/usr/bin/env python

import argparse
import numpy as np
from chemsnail.classes import system

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported vib input into cpmd trajectory", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xvibs,...)")
    parser.add_argument("-f", help="Output file name (standard: 'TRAJSAVED')", default='TRAJSAVED')
    parser.add_argument("-mw", action='store_true', help="Assume mass-weighed coordinates in fn (default: false).", default=False)
    parser.add_argument("-ignore_warnings", action='store_true', help="Ignore warnings regarding orthonormality (default: false).", default=False)
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-modelist", nargs='+', help="List of modes  (0-based index, default: all).", default=[None])
    parser.add_argument("-cell_aa", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: 0 0 0 90 90 90).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-pp", help="Pseudopotential (default: 'MT_BLYP')", default='MT_BLYP')
    parser.add_argument("-bs", help="Method for treatment of the non-local part of the pp (default: '')", default='')
    parser.add_argument("-factor", help="Velocity scaling, good for inversion (default: 1.0).", default=1.0)

    args = parser.parse_args()
    args.cell_aa = np.array(args.cell_aa).astype(float)
    args.factor = float(args.factor)
    if not any(args.modelist):
        system.Molecule(fmt='xvibs',
                          **vars(args)
                          ).Modes.write_nuclear_velocities(args.f,fmt='cpmd',pp=args.pp,bs=args.bs,factor=args.factor)
    else:
        system.Molecule(fmt='xvibs',
                          **vars(args)
                          ).Modes.write_nuclear_velocities(args.f,fmt='cpmd',modelist=[int(m) for m in args.modelist],pp=args.pp,bs=args.bs,factor=args.factor)


if(__name__ == "__main__"):
    main()
