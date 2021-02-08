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

from chirpy.create.moments import OriginGauge
from chirpy.classes import system, trajectory
from chirpy.topology import mapping
from chirpy.physics import constants
from chirpy.interface import cpmd
from chirpy import config


def main():
    parser = argparse.ArgumentParser(
            description="Process MOMENTS output of electronic (Wannier)\
                         states and add (classical) nuclear contributions to\
                         generate molecular moments based on a given\
                         topology.\
                         Supports only CPMD input and output.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "TOPOLOGY",
            help="pdb file with topology including cell specifications"
            )
    parser.add_argument(
            "TRAJECTORY",
            help="TRAJECTORY"
            )
    parser.add_argument(
            "MOMENTS",
            help="MOMENTS"
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Frame range for reading files.",
            default=None,
            type=int,
            )
#     parser.add_argument(
#             "--hag",
#             action='store_true',
#             default=False,
#             help="Use heavy atom gauge."
#             )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='MOL'
            )
    parser.add_argument(
            "--verbose",
            action='store_true',
            help="Print info and progress.",
            default=False,
            )

    args = parser.parse_args()
    if args.range is None:
        args.range = (0, 1, float('inf'))
    config.set_verbose(args.verbose)

    _traj = system.Supercell(args.TRAJECTORY,
                             fmt='cpmd',
                             range=args.range,
                             fn_topo=args.TOPOLOGY,
                             # --- this is costly depending on no of mols
                             wrap_molecules=True,
                             )

    _moms_e = trajectory.MOMENTS(args.MOMENTS,
                                 fmt='cpmd',
                                 range=args.range,
                                 )
    # -- ToDo: Add test for neutrality of charge
    if (_total_charge := _moms_e.n_atoms * (-2) +
            constants.symbols_to_valence_charges(_traj.symbols).sum()) != 0.0:
        warnings.warn(f'Got non-zero cell charge {_total_charge}!',
                      RuntimeWarning, stacklevel=2)

    n_map = np.array(_traj.mol_map)

    _cell = _traj.cell_aa_deg
    # _cell[:3] *= constants.l_aa2au

    for _iframe, (_p_fr, _m_fr) in enumerate(zip(_traj.XYZ, _moms_e)):
        # --- generate classical nuclear moments
        gauge_n = OriginGauge(
               trajectory.MOMENTSFrame.from_classical_nuclei(_traj.XYZ._frame),
               cell=_cell
               )

        # --- load Wannier data and get nearest atom assignment
        gauge_e = OriginGauge(_moms_e, cell=_cell)
        e_map = n_map[mapping.nearest_neighbour(gauge_e.r_au*constants.l_au2aa,
                                                gauge_n.r_au*constants.l_au2aa,
                                                cell=_cell)]

        # --- combine nuclear and electronic contributions
        gauge = gauge_e + gauge_n
        assignment = np.concatenate((e_map, n_map))

        # --- switch to molecular origins
        _com = _traj.XYZ.mol_com_aa  # * constants.l_aa2au
        gauge.switch_origin_gauge(_com, assignment)

        # --- write output
        append = False
        if _iframe > 0:
            append = True

        cpmd.cpmdWriter(
                 args.f,
                 np.array([np.concatenate((gauge.r_au*constants.l_au2aa,
                                           gauge.c_au,
                                           gauge.m_au), axis=-1)]),
                 frame=_iframe,
                 append=append,
                 write_atoms=False)


if __name__ == "__main__":
    main()
