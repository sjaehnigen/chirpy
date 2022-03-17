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

import argparse
import numpy as np
import warnings

from chirpy.create.moments import OriginGauge
from chirpy.classes import system, trajectory
from chirpy.topology import mapping
from chirpy import constants
from chirpy.physics import classical_electrodynamics as ed
from chirpy.interface import cpmd
from chirpy import config


def main():
    parser = argparse.ArgumentParser(
            description="Process MOMENTS output of electronic (Wannier)\
                         states and add (classical) nuclear contributions to\
                         generate molecular moments based on a given\
                         topology.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "TOPOLOGY",
            help="PDB file with topology including cell specifications"
            )
    parser.add_argument(
            "TRAJECTORY",
            help="(classical) nuclear part with positions and velocities."
            )
    parser.add_argument(
            "MOMENTS",
            help="(quantum) electronic part with localised (Wannier) gauge "
                 "positions, current dipole, and magnetic dipole moments."
            )
    parser.add_argument(
            "--T_format",
            help="File format of TRAJECTORY (e.g. xyz, tinker, cpmd)",
            default='cpmd',
            )
    parser.add_argument(
            "--M_format",
            help="File format of MOMENTS (e.g. cpmd, tinker)",
            default='cpmd',
            )
    parser.add_argument(
            "--T_units",
            help="Column units of TRAJECTORY.",
            default='default',
            )
    parser.add_argument(
            "--M_units",
            help="Column units of MOMENTS.",
            default='default',
            )
    parser.add_argument(
            "--electronic_centers",
            help="(Wannier) centers of charge of electronic part "
                 "(for better assignment of electrons and --position_form)."
            )
    parser.add_argument(
            "--EC_format",
            help="File format of --electronic_centers (e.g. cpmd, tinker)",
            default='cpmd',
            )
    parser.add_argument(
            "--EC_units",
            help="Column units of --electronic_centers.",
            default='default',
            )
    parser.add_argument(
            "--position_form",
            action='store_true',
            help="also compute the electric dipole moment and add it to "
                 "the output file.",
            default=False,
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Frame range for reading files.",
            default=None,
            type=int,
            )
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

    SYS = system.Supercell(args.TRAJECTORY,
                           fmt=args.T_format,
                           range=args.range,
                           units=args.T_units,
                           fn_topo=args.TOPOLOGY,
                           # --- generate mol centers, costly
                           wrap_molecules=True,
                           )
    NUC = SYS.XYZ

    ELE = trajectory.MOMENTS(args.MOMENTS,
                             fmt=args.M_format,
                             range=args.range,
                             units=args.M_units,
                             )
    _trajectory = (NUC, ELE)
    if args.electronic_centers is not None:
        WC = trajectory.XYZ(args.electronic_centers,
                            fmt=args.EC_format,
                            range=args.range,
                            units=args.EC_units,
                            # fn_topo=args.TOPOLOGY,
                            )
        _trajectory += (WC,)

    # --- test for neutrality of charge
    if (_total_charge := ELE.n_atoms * (-2) +
            constants.symbols_to_valence_charges(NUC.symbols).sum()) != 0.0:
        warnings.warn(f'Got non-zero cell charge {_total_charge}!',
                      config.ChirPyWarning, stacklevel=2)

    n_map = np.array(SYS.mol_map)
    _cell = SYS.cell_aa_deg

    for _iframe, (_frame) in enumerate(zip(*_trajectory)):

        # --- generate classical nuclear moments
        Qn_au = constants.symbols_to_valence_charges(NUC.symbols)
        gauge_n = OriginGauge(
               origin_aa=NUC.pos_aa,
               current_dipole_au=ed.current_dipole_moment(NUC.vel_au, Qn_au),
               magnetic_dipole_au=np.zeros_like(NUC.vel_au),
               charge_au=Qn_au,
               cell_aa_deg=_cell,
               )

        # --- load Wannier data and get nearest atom assignment
        gauge_e = OriginGauge(
               origin_aa=ELE.pos_aa,
               current_dipole_au=ELE.c_au,
               magnetic_dipole_au=ELE.m_au,
               charge_au=-2,
               cell_aa_deg=_cell,
               )

        # --- shift gauge to electric centers (optional)
        if args.electronic_centers is not None:
            gauge_e.shift_origin_gauge(WC.pos_aa)
        if args.position_form:
            if args.M_format != 'cpmd':
                warnings.warn('assuming valence charges for atoms. No core '
                              'electrons considered.',
                              stacklevel=2)
            for _gauge in [gauge_e, gauge_n]:
                _gauge.d_au = np.zeros_like(_gauge.c_au)
                _gauge._set += 'd'

        e_map = n_map[mapping.nearest_neighbour(gauge_e.r_au*constants.l_au2aa,
                                                gauge_n.r_au*constants.l_au2aa,
                                                cell=_cell)]

        # --- combine nuclear and electronic contributions
        gauge = gauge_e + gauge_n
        assignment = np.concatenate((e_map, n_map))

        # --- shift to molecular origins
        _com = NUC.mol_com_aa
        gauge.shift_origin_gauge(_com, assignment)

        # --- test for neutrality of charge
        if np.any((_mol := gauge.q_au != 0.0)):
            warnings.warn('Got non-zero charge for molecules '
                          f'{np.where(_mol)[0]}: {gauge.q_au[_mol]}',
                          config.ChirPyWarning, stacklevel=2)

        # --- write output
        append = False
        if _iframe > 0:
            append = True

        if args.position_form:
            _data = np.array([np.concatenate((gauge.r_au*constants.l_au2aa,
                                              gauge.c_au,
                                              gauge.m_au,
                                              gauge.d_au), axis=-1)])
        else:
            _data = np.array([np.concatenate((gauge.r_au*constants.l_au2aa,
                                              gauge.c_au,
                                              gauge.m_au), axis=-1)])

        cpmd.cpmdWriter(
                 args.f,
                 _data,
                 frame=_iframe,
                 append=append,
                 write_atoms=False)


if __name__ == "__main__":
    main()
