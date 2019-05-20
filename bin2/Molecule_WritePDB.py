#!/usr/bin/env python3

import argparse
import numpy as np 
from classes import system
from topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported input into XYZ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xyz.pdb,xvibs,...)")
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-cell_aa", nargs=6, help="Orthorhombic cell parametres a b c al be ga in angstrom/degree (default: None).", default=[0.0,0.0,0.0,90.,90.,90.])
    parser.add_argument("-f", help="Output file name", default='out.pdb')
    args = parser.parse_args()
    if args.cell_aa is not None: args.cell_aa = np.array(args.cell_aa).astype(float)

    #quick workaround (molecule does not yet have a write routine)
    system.Molecule(**vars(args),wrap_mols=True).XYZData.write(args.f,fmt='pdb')
    #system.Supercell(**vars(args),wrap_mols=True).XYZData.write(args.f,fmt='pdb')


if(__name__ == "__main__"):
    main()
