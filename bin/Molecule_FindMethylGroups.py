#!/usr/bin/env python

import argparse
from chemsnail.classes import system
from chemsnail.topology import symmetry

def main():
    '''Unit Cell parametres are taken from fn if needed'''
    parser=argparse.ArgumentParser(description="Find Methyl Group (like) subgroups in molecule", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn", help="file (xy,pdb,xvibs,...)")
    parser.add_argument("-hetatm", action='store_true', help="Also include heteroatoms into search (e.g. finding NH3+ groups; default: False)", default=False)
    args = parser.parse_args()
    symmetry.find_methyl_groups(system.Molecule(args.fn),hetatm=bool(args.hetatm))

if(__name__ == "__main__"):
    main()