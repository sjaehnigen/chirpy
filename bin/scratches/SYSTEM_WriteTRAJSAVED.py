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


def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser=argparse.ArgumentParser(description="Convert any supported input into XYZ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            "fn",
            help="file (xyz.pdb,xvibs,...)"
            )
    parser.add_argument(
            "--fn_topo",
            help="file (xyz.pdb,xvibs,...)",
            default=None
            )
    parser.add_argument(
            "--cell_aa",
            nargs=6,
            help="Cell specification in angstrom and degree (optional)."
            )
    parser.add_argument("--center_coords",
                        nargs='+',
                        help="Center atom list (id starting from 0) in cell \
                                and wrap",
                        default=None,
                        type=int,
                        )

    parser.add_argument(
            "--center_residue",
            help="Center Residue in cell centre and wrap molecules (overrides center_coords option;\
            default: false; cell_aa parametre overrides default origin).",
            type=int,
            default=-1
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='TRAJSAVED'
            )
    parser.add_argument(
            "-pp",
            help="Pseudopotential",
            default='MT_BLYP'
            )
    parser.add_argument(
            "-bs",
            help="Method for treatment of the non-local part of the pp",
            default=''
            )
    parser.add_argument(
            "--factor",
            help="Velocity scaling, good for inversion",
            default=1.0
            )
    args = parser.parse_args()

    if args.center_residue != -1 and args.fn_topo is None:
        raise Exception('ERROR: center_residue needs topology file (fn_topo)!')
    if args.cell_aa is None:
        del args.cell_aa
    else:
        print('Override internal cell specifications by input.')
        args.cell_aa = np.array(args.cell_aa).astype(float)

    largs = vars(args)
    _load = system.Supercell(args.fn, **largs)

    # python3.8: use walrus
    center_coords = largs.pop('center_coords')
    # align_coords = largs.pop('align_coords')
    if center_coords is not None:
        _load.XYZ.center_coordinates(center_coords, **largs)
    # if align_coords is not None:
    #     _load.XYZ.align_coordinates(align_coords, **largs)

    _load.write(
            args.f,
            fmt='cpmd',
            pp=args.pp,
            bs=args.bs,
            factor=args.factor
            )


if(__name__ == "__main__"):
    main()
