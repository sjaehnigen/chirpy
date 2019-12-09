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
from chirpy.classes import system


def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser = argparse.ArgumentParser(
            description="Wrap atoms from XYZ file into PBC box",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn1", help="file 1 (xy,pdb,xvibs,...)")
    parser.add_argument("fn2", help="file 2 (xy,pdb,xvibs,...)")
    args = parser.parse_args()
    fn1 = args.fn1
    fn2 = args.fn2
    mol1 = system.Supercell(fn1)
    mol2 = system.Supercell(fn2)

    for _fr in mol1.XYZ:
        print('Frame:', _fr)
        assign = mol1.XYZ.map_frame(mol1.XYZ, mol2.XYZ)
        outbuf = ['%35d -------> %3d' % (i+1, j+1)
                  for i, j in enumerate(assign)]
        print('%35s          %s' % (fn1, fn2))
        print('\n'.join(outbuf))


if(__name__ == "__main__"):
    main()
