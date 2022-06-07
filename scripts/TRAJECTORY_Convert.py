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
#  Copyright (c) 2010-2022, The ChirPy Developers.
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

import sys
import argparse
import numpy as np
import warnings

from chirpy.classes import system, trajectory
from chirpy import config


def main():
    '''Convert and process trajectory'''
    parser = argparse.ArgumentParser(
            description="Convert and process trajectory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input trajectory file."
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. xyz, pdb, cpmd; optional).",
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
            "--units",
            help="List of chirpy unit tags for data columns in ALL files, "
                 "e.g., (\'length\', \'aa\')."
                 "\'default\' refers to the ChirPy unit convention for each "
                 "file format",
            default='default',
            nargs='+',
            )
    parser.add_argument(
            "--mask_frames",
            nargs='+',
            help="If True: Check for duplicate frames (based on comment line) "
                 "and skip them (may take some time).\n"
                 "If list of arguments: Skip given frames without further "
                 "checking (fast).",
            default=None
            )
    parser.add_argument(
            "--fn_topo",
            help="Topology file containing metadata (cell, \
                    molecules, ...).",
            default=None,
            )
    parser.add_argument(
            "--fn_vel",
            help="Additional trajectory file with velocities (optional). "
                 "Assumes atomic units.",
            default=None,
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
            type=float,
            default=None,
            )
    parser.add_argument(
            "--wrap",
            action='store_true',
            help="Wrap atoms into cell.",
            default=False
            )
    parser.add_argument(
            "--wrap_molecules",
            action='store_true',
            help="Wrap molecules into cell (requires topology).",
            default=False
            )
    parser.add_argument(
            "--center_coords",
            nargs='+',
            help="Center atom list (id starting from 0) in cell \
                    and wrap or True for selecting all atoms.",
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
            "--align_coords",
            nargs='+',
            help="Align atom list (id starting from 0) or True \
                    for selecting all atoms.",
            default=False,
            )
    parser.add_argument(
            "--force_centering",
            action='store_true',
            help="Enforce centering after alignment.",
            default=False,
            )
    parser.add_argument(
            "--weights",
            help="Atom weights used for atom alignment, centering, wrapping"
                 " (\'masses\' or \'one\')",
            default='masses'
            )
    parser.add_argument(
            "--select_molecules",
            nargs='+',
            help="Write only coordinates of given molecular ids starting from"
                 " 0 (requires a topology file). Atoms are arranged in "
                 "molecular blocks. "
                 "Combination with --select_atoms and --select_elements "
                 "returns the intersecting set, but destroys order "
                 "and index repetitions.",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--select_atoms",
            nargs='+',
            help="Write only coordinates of given atoms (id starting from 0)."
                 "Keeps atom order. "
                 "Combination with --select_molecules and --select_elements "
                 "returns the intersecting set, but destroys order "
                 "and index repetitions.",
            default=None,
            )
    parser.add_argument(
            "--select_elements",
            nargs='+',
            help="Write only coordinates of given elements (through symbols)."
                 "Keeps atom order. "
                 "Combination with --select_atoms and --select_molecules "
                 "returns the intersecting set, but destroys order "
                 "and index repetitions.",
            default=None,
            )
    parser.add_argument(
            "--sort",
            action='store_true',
            help="Alphabetically sort atoms",
            default=False
            )
    parser.add_argument(
        "--multiply",
        nargs=3,
        help="Repeat cell in X, Y, Z.",
        default=None,
        type=int
        )

    parser.add_argument(
            "--convert_to_moments",
            action='store_true',
            help="Write classical electro-magnetic moments instead.",
            default=False,
            )
    parser.add_argument(
            "--outputfile", "-o", "-f",
            help="Output file name",
            default='out.xyz'
            )
    parser.add_argument(
            "--output_format",
            help="Output file format (e.g. xyz, pdb, cpmd; optional).",
            default=None,
            )
    parser.add_argument(
            "--write_atoms",
            help="Print ATOMS and XYZ topology (--output_format cpmd only).",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--pp",
            help="pseudopotential for --print_atoms.",
            default='MT_BLYP KLEINMAN-BYLANDER',
            )
    parser.add_argument(
            "--verbose",
            action='store_true',
            help="Print info and progress.",
            default=False,
            )
    args = parser.parse_args()

    config.set_verbose(args.verbose)
    # --------------------------------------------------------
    # --- parse and combine arguments: bash ---> chirpy
    if bool(args.center_coords):
        # --- ToDo: workaround
        if args.center_coords[0] == 'True':
            args.center_coords = True
        elif args.center_coords[0] == 'False':
            args.center_coords = False
        else:
            args.center_coords = [int(_a) for _a in args.center_coords]

    if bool(args.align_coords):
        if args.align_coords[0] == 'True':
            args.align_coords = True
        elif args.align_coords[0] == 'False':
            args.align_coords = False
        else:
            args.align_coords = [int(_a) for _a in args.align_coords]

        if bool(args.center_coords) or args.center_molecule is not None:
            warnings.warn('Using centering/wrapping and aligning in one call '
                          'may not yield the desired result (use two '
                          'consecutive calls if this is the case).',
                          config.ChirPyWarning,
                          stacklevel=2)

    # --- delete empty arguments
    _guess_mol_map = False
    if args.fn_topo is None:
        if args.select_molecules is not None:
            warnings.warn('expected topology file for --select_molecules and '
                          'continues with automatic guess of molecular map',
                          config.ChirPyWarning,
                          stacklevel=2)
        _guess_mol_map = True
        del args.fn_topo

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)

    if args.range is None:
        del args.range

    # --- set defaults for format
    i_fmt = args.input_format
    o_fmt = args.output_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
    if o_fmt is None:
        if args.convert_to_moments and args.outputfile in ['MOMENTS', 'MOL',
                                                           'ATOM']:
            o_fmt = 'cpmd'
        elif args.outputfile in ['TRAJSAVED', 'TRAJECTORY', 'CENTERS']:
            o_fmt = 'cpmd'
        else:
            o_fmt = args.outputfile.split('.')[-1].lower()
    elif o_fmt == 'tinker':
        o_fmt = 'arc'
    if args.outputfile == 'out.xyz':
        args.outputfile = 'out.' + o_fmt

    # --------------------------------------------------------
    # --- ToDo: Caution when passing all arguments to object!
    largs = vars(args)

    skip = largs.pop('mask_frames')
    select_molecules = largs.pop('select_molecules')
    select_atoms = largs.pop('select_atoms')
    select_elements = largs.pop('select_elements')
    multiply = largs.pop('multiply')

    _files = [args.fn]
    if args.fn_vel is not None:
        _files.append(args.fn_vel)

    if skip is None:
        _load = system.Supercell(*_files, fmt=i_fmt, **largs)
        largs.update({'skip': []})
    else:
        if skip[0] in ['True', 'False']:
            if skip[0] == 'True':
                skip = bool(skip[0])
                _load = system.Supercell(*_files, fmt=i_fmt, **largs)
                skip = _load.XYZ.mask_duplicate_frames()
                largs.update({'skip': skip})
            else:
                _load = system.Supercell(*_files, fmt=i_fmt, **largs)
                largs.update({'skip': []})
        else:
            skip = [int(_a) for _a in skip]
            largs.update({'skip': skip})
            _load = system.Supercell(*_files, fmt=i_fmt, **largs)

    if multiply is not None:
        _load.XYZ.repeat(tuple(multiply))

    # --- object ----> file
    if args.outputfile not in ['None', 'False']:

        # --- parse selection
        _selection = []
        if select_atoms is not None:
            _selection = select_atoms
        if select_molecules is not None:
            if _guess_mol_map:
                _load.define_molecules()
            __selection = [_i
                           for _m in select_molecules
                           for _i, _s in enumerate(_load.mol_map)
                           if _s == _m]
            if _selection == []:
                _selection = __selection
            else:
                _selection = list(set(_selection) & set(__selection))
        if select_elements is not None:
            __selection = np.where([_s in select_elements
                                    for _s in _load.symbols])[0].tolist()
            if _selection == []:
                _selection = __selection
            else:
                _selection = list(set(_selection) & set(__selection))
        if _selection == []:
            _selection = None
        # ----

        if args.verbose:
            print('Writing output...', file=sys.stderr)

        if args.convert_to_moments:
            for _iframe, _p_fr in enumerate(_load.XYZ):
                _moment = trajectory.MOMENTSFrame.from_classical_nuclei(
                        _load.XYZ._frame)
                # --- write output
                append = False
                if _iframe > 0:
                    append = True
                largs = {'append': append, 'frame': _iframe}
                _moment.write(args.outputfile, fmt=o_fmt, **largs)

        else:
            largs = {}
            if o_fmt == 'cpmd':
                largs = dict(
                        pp=args.pp,
                        write_atoms=args.write_atoms,
                        )
            _load.write(args.outputfile, fmt=o_fmt, rewind=False,
                        selection=_selection, **largs)


if __name__ == "__main__":
    main()
