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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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
import numpy as np
import warnings

from chirpy.classes import system, trajectory


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
                 "Assumes atomic units. BETA",
            default=None,
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
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
            "--weight",
            help="Atom weights used for atom alignment, centering, wrapping",
            default='mass'
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
            "--sort",
            action='store_true',
            help="Alphabetically sort atoms",
            default=False
            )
    parser.add_argument(
            "--convert_to_moments",
            action='store_true',
            help="Write classical electro-magnetic moments instead.",
            default=False,
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='out.xyz'
            )
    parser.add_argument(
            "--output_format",
            help="Output file format (e.g. xyz, pdb, cpmd; optional).",
            default=None,
            )
    parser.add_argument(
            "--pp",
            help="Pseudopotential (for CPMD ATOMS section only).",
            default='MT_BLYP KLEINMAN-BYLANDER',
            )
    args = parser.parse_args()

    # --- ToDo: workaround
    if bool(args.center_coords):
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

    if args.fn_topo is None:
        del args.fn_topo

    if args.pp is None:
        del args.pp

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    if args.range is None:
        args.range = (0, 1, float('inf'))

    i_fmt = args.input_format
    o_fmt = args.output_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
    if o_fmt is None:
        if args.convert_to_moments and args.f in ['MOMENTS', 'MOL', 'ATOM']:
            o_fmt = 'cpmd'
        elif args.f in ['TRAJSAVED', 'TRAJECTORY']:
            o_fmt = 'cpmd'
        else:
            o_fmt = args.f.split('.')[-1].lower()
    elif o_fmt == 'tinker':
        o_fmt = 'arc'
    if args.f == 'out.xyz':
        args.f = 'out.' + o_fmt

    # --- Caution when passing all arguments to object!
    largs = vars(args)

    if bool(args.align_coords):
        if bool(args.center_coords) or args.center_molecule is not None:
            warnings.warn('Using centering/wrapping and aligning in one call '
                          'may not yield the desired result (use two '
                          'consecutive calls if this is the case).',
                          stacklevel=2)

    skip = largs.pop('mask_frames')
    if skip is not None:
        if skip[0] in ['True', 'False']:
            if skip[0] == 'True':
                skip = bool(skip[0])
                _load = system.Supercell(args.fn, fmt=i_fmt, **largs)
                skip = _load.XYZ.mask_duplicate_frames(verbose=False)
                largs.update({'skip': skip})
            else:
                _load = system.Supercell(args.fn, fmt=i_fmt, **largs)
                largs.update({'skip': []})
        else:
            skip = [int(_a) for _a in skip]
            largs.update({'skip': skip})
            _load = system.Supercell(args.fn, fmt=i_fmt, **largs)
    else:
        _load = system.Supercell(args.fn, fmt=i_fmt, **largs)
        largs.update({'skip': []})

    if args.fn_vel is not None:
        nargs = {}
        for _a in [
            'range',
            'fn_topo',
            'sort',
            'skip',
                   ]:
            nargs[_a] = largs.get(_a)

        if i_fmt in ['tinker', 'arc', 'vel']:
            nargs['units'] = ('velocity', 'aaperfs')

        _load_vel = system.Supercell(args.fn_vel, fmt=i_fmt, **nargs)
        _load.XYZ.merge(_load_vel.XYZ, axis=-1)

        # --- repeat alignment call after merge to include velocities into
        #     iterator mask
        # ToDo: (IS THIS NECESSARY?)
        if bool(args.align_coords):
            _load.XYZ.align_coordinates(
                    selection=args.align_coords,
                    weight=args.weight,
                    force_centering=args.force_centering,
                    align_ref=_load.XYZ._frame._align_ref)

    extract_molecules = largs.pop('extract_molecules')

    if extract_molecules is not None:
        _load.extract_molecules(extract_molecules)

    if args.f not in ['None', 'False']:
        if args.convert_to_moments:
            # --- BETA
            for _iframe, _p_fr in enumerate(_load.XYZ):
                _moment = trajectory.MOMENTSFrame.from_classical_nuclei(
                        _load.XYZ._frame)
                # --- write output
                append = False
                if _iframe > 0:
                    append = True
                largs = {'append': append, 'frame': _iframe}
                _moment.write(args.f, fmt=o_fmt, **largs)

                # cpmd.cpmdWriter(
                #  args.f,
                #  np.array([np.concatenate((gauge.r_au*constants.l_au2aa,
                #                            gauge.c_au,
                #                            gauge.m_au), axis=-1)]),
                #  frame=_iframe,
                #  append=append,
                #  write_atoms=False)

        else:
            largs = {}
            if 'pp' in args:
                largs = {'pp': args.pp}
            _load.write(args.f, fmt=o_fmt, rewind=False, **largs)


if __name__ == "__main__":
    main()
