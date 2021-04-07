#!/usr/bin/env python
# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import argparse
from chirpy.classes import system


def main():
    '''Create a topology file from input.'''
    parser = argparse.ArgumentParser(
        description="Create a topology file from input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
            "fn",
            help="Input structure file (xyz.pdb,xvibs,...)"
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. xyz, pdb, cpmd; optional).",
            default=None,
            )
    parser.add_argument(
            "--center_coords",
            nargs='+',
            help="Center atom list (id starting from 0) in cell \
                    and wrap or \'True\' for selecting all atoms.",
            default=False,
            )
    parser.add_argument(
            "--center_molecule",
            help="Center residue (resid as given in fn_topo) in \
                    cell and wrap (requires a topology file).",
            default=None,
            type=int,
            )

    parser.add_argument(
            "--weight",
            help="Atom weights used for centering and wrapping of molecules",
            default='mass'
            )
    parser.add_argument(
        "--cell_aa_deg",
        nargs=6,
        help="Orthorhombic cell parametres a b c al be ga in angstrom/degree.",
        type=float,
        default=None
        )
    parser.add_argument(
            "--keep_molecules",
            action='store_true',
            help="Do not re-write molecular assignment.",
            default=False
            )
    parser.add_argument(
            "--keep_positions",
            action='store_true',
            help="Do not modifiy positions at all (wrap, center, ...).",
            default=False
            )
    parser.add_argument(
            "--wrap_molecules",
            action='store_true',
            help="Wrap molecules instead of atoms into cell.",
            default=False
            )
    parser.add_argument("-f", help="Output file name", default='out.pdb')
    args = parser.parse_args()
    if bool(args.center_coords):
        if args.center_coords[0] == 'True':
            args.center_coords = True
        elif args.center_coords[0] == 'False':
            args.center_coords = False
        else:
            args.center_coords = [int(_a) for _a in args.center_coords]

    if args.keep_positions:
        args.center_coords = False
        args.center_molecule = None
        args.wrap_molecules = False

    i_fmt = args.input_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
    if args.cell_aa_deg is None:
        del args.cell_aa_deg

    _load = system.Molecule(**vars(args), fmt=i_fmt)
    _load.XYZ._check_distances()

    # --- keep only coordinates
    if hasattr(_load, 'Modes'):
        del _load.Modes

    if not args.keep_molecules:
        _load.define_molecules()

    if args.wrap_molecules:
        _load.wrap_molecules()  # algorithm='heavy_atom')
    elif not args.keep_positions:
        _load.wrap()
    _load.write(args.f)


if(__name__ == "__main__"):
    main()
