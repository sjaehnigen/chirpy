#!/usr/bin/env python

import argparse
import numpy as np 
from classes import system
from topology import mapping

def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported input into XYZ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn",       help="coord file (xyz.pdb,xvibs,...)")
    parser.add_argument("multiply", help="number of replica per dimension (three numbers)", nargs=3)
    parser.add_argument("-abc",     help="cell parametres a, b, and c in angstrom (overwrites what is given in fn)", nargs=3, default=None)
    parser.add_argument("-albega",  help="cell parametres alpha, beta, and gamma in degree (overwrites what is given in fn)", nargs=3, default=None)
    parser.add_argument("-f",       help="Output file name (standard: 'out.xyz')", default='out.xyz')
    args = parser.parse_args()

    if getattr(args,'abc') is not None:
        print('I use user-defined cell.')
        if getattr(args,'albega') is not None:
            cell_aa = np.concatenate((np.array(args.abc).astype(float),np.array(args.albega).astype(float)))
        else:
            cell_aa = np.concatenate((np.array(args.abc).astype(float),np.ones((3))*90.))
        print(cell_aa)
        system.Molecule(args.fn,cell_aa=cell_aa,cell_multiply=np.array(args.multiply).astype(int),wrap_mols=True).XYZData.write(args.f)
    else:
        system.Molecule(args.fn,cell_multiply=np.array(args.multiply).astype(int),wrap_mols=True).XYZData.write(args.f,attr='data')
#    system = Molecule(fn_modes,fmt='xvibs',cell_aa=cell_aa,mw=True,ignore_warnings=True)



if(__name__ == "__main__"):
    main()
