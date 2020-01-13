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
import warnings

from chirpy.classes import system


def main():
    '''Convert and process trajectory'''
    parser = argparse.ArgumentParser(
            description="Convert and process trajectory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument(
            "--fn_vel",
            help="Additional trajectory file with velocities (optional). "
                 "Assumes atomic units. BETA",
            default=None,
            )

    parser.add_argument(
            "--center_coords",
            nargs='+',
            help="Center atom list (id starting from 0) in cell \
                    and wrap or \'True\' for selecting all atoms.",
            default=None,
            )
    parser.add_argument(
            "--center_molecule",
            help="Center residue (resid as given in fn_topo) in \
                    cell and wrap (requires a topology file).",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--align_coords",
            nargs='+',
            help="Align atom list (id starting from 0)  or \'True\' \
                    for selecting all atoms.",
            default=None,
            )
    parser.add_argument(
            "--force_centre",
            action='store_true',
            help="Enforce centering after alignment.",
            default=False,
            )
    parser.add_argument(
            "--use_com",
            action='store_true',
            help="Use centre of mass instead of centre of geometry \
                    as reference",
            default=False
            )
    parser.add_argument(
            "--wrap",
            action='store_true',
            help="Wrap atoms in cell.",
            default=False
            )
    parser.add_argument(
            "--wrap_molecules",
            action='store_true',
            help="Wrap molecules in cell (requires topology).",
            default=False
            )
    parser.add_argument(
            "--extract_molecules",
            nargs='+',
            help="Write only coordinates of given molecular ids starting from 0 \
                    (requires a topology file).",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
            default=None,
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Range of frames to read (start, step, stop)",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--fn_topo",
            help="Topology file containing metadata (cell, \
                    molecules, ...).",
            default=None,
            )
    parser.add_argument(
            "--sort",
            action='store_true',
            help="Alphabetically sort atoms",
            default=False
            )
    parser.add_argument(
            "-f",
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

    if args.fn_vel is not None:
        warnings.warn("External velocity file not tested for all options!",
                      stacklevel=2)
        nargs = {}
        for _a in [
            'range',
            'fn_topo',
            'sort',
                   ]:
            nargs[_a] = largs.get(_a)

        _load_vel = system.Supercell(args.fn_vel, **nargs)
        _load.XYZ.merge(_load_vel.XYZ, axis=-1)

    # python3.8: use walrus
    center_coords = largs.pop('center_coords')
    align_coords = largs.pop('align_coords')
    extract_molecules = largs.pop('extract_molecules')

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

    if extract_molecules is not None:
        _load.extract_molecules(extract_molecules)

    _load.write(args.f, fmt='xyz', rewind=False)


if __name__ == "__main__":
    main()
