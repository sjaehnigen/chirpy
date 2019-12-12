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
    '''Convert any supported input into XYZ format'''
    parser = argparse.ArgumentParser(
            description="Convert any supported input into XYZ format",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--center_coords",
                        nargs='+',
                        help="Center atom list (id starting from 0) in cell \
                                and wrap or \'True\' for selecting all atoms.",
                        default=None,
                        )
    parser.add_argument("--align_coords",
                        nargs='+',
                        help="Align atom list (id starting from 0)  or \'True\' \
                                for selecting all atoms.",
                        default=None,
                        )
    parser.add_argument("--force_centre",
                        action='store_true',
                        help="Enforce centering after alignment.",
                        default=False,
                        )
    parser.add_argument("--use_com",
                        action='store_true',
                        help="Use centre of mass instead of centre of geometry \
                                as reference",
                        default=False
                        )
    parser.add_argument("--wrap",
                        action='store_true',
                        help="Wrap atoms in cell.",
                        default=False
                        )
    parser.add_argument("--cell_aa_deg",
                        nargs=6,
                        help="Use custom cell parametres a b c al be ga in \
                                angstrom/degree",
                        default=None,
                        )
    parser.add_argument("--range",
                        nargs=3,
                        help="Range of frames to read (start, step, stop)",
                        default=None,
                        type=int,
                        )
    parser.add_argument("--fn_topo",
                        help="Topology file containing metadata (cell, \
                                molecules, ...).",
                        default=None,
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )
    args = parser.parse_args()

    if args.fn_topo is None:
        del args.fn_topo

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    if args.range is None:
        args.range = (0, 1, float('inf'))

    largs = vars(args)
    _load = system.Supercell(args.fn, **largs)

    # python3.8: use walrus
    center_coords = largs.pop('center_coords')
    align_coords = largs.pop('align_coords')

    if center_coords is not None:
        if center_coords[0] in ['True', 'False']:
            center_coords = bool(center_coords[0])
        else:
            center_coords = [int(_a) for _a in center_coords]
        _load.XYZ.center_coordinates(center_coords, **largs)

    if align_coords is not None:
        if align_coords[0] in ['True', 'False']:
            align_coords = bool(align_coords[0])
        else:
            align_coords = [int(_a) for _a in align_coords]
        _load.XYZ.align_coordinates(align_coords, **largs)

    _load.write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
