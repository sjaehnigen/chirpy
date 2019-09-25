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
    parser=argparse.ArgumentParser(description="Convert any supported input into XYZ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xyz.pdb,xvibs,...)")
    parser.add_argument("-fn_topo", help="file (xyz.pdb,xvibs,...)", default=None)
    parser.add_argument("-cell_aa", nargs=6, help="Cell specification in angstrom and degree (optional).")
    parser.add_argument("-center_coords", action='store_true', help="Center Coordinates in cell centre or at origin (default: false; box_aa parametre overrides default origin).", default=False)
    parser.add_argument("-center_residue", help="Center Residue in cell centre and wrap molecules (overrides center_coords option; default: false; box_aa parametre overrides default origin).", type=int, default=-1)
    parser.add_argument("-f", help="Output file name (standard: 'TRAJSAVED')", default='TRAJSAVED')
    parser.add_argument("-pp", help="Pseudopotential (default: 'MT_BLYP')", default='MT_BLYP')
    parser.add_argument("-bs", help="Method for treatment of the non-local part of the pp (default: '')", default='')
    parser.add_argument("-factor", help="Velocity scaling, good for inversion (default: 1.0).", default=1.0)
    args = parser.parse_args()

    if args.center_residue!=-1 and args.fn_topo is None: raise Exception('ERROR: center_residue needs topology file (fn_topo)!')
    if args.cell_aa is None: del args.cell_aa
    else: 
        print('Override internal cell specifications by input.')
        args.cell_aa = np.array(args.cell_aa).astype(float)

    #system.Molecule(**vars(args)).XYZData.write(args.f,fmt='cpmd',pp=args.pp,bs=args.bs,factor=args.factor)
    system.Supercell(**vars(args)).XYZData.write(args.f,fmt='cpmd',pp=args.pp,bs=args.bs,factor=args.factor)


if(__name__ == "__main__"):
    main()
