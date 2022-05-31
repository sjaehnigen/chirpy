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
import tqdm

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
            help="Frame range for reading files. Frame numbers are not "
                 "preserved in output",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--batch_size",
            help="No. of frames processed at once. Needs to be reduced for "
                 "very large molecules or low memory availability.",
            default=1000000,
            type=int,
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='MOL'
            )
    parser.add_argument(
            "--do_not_join",
            action='store_true',
            help="Do not join molecules before computing gauge to accelerate. "
                 "Enable ONLY if molecules are not broken across boundaries.",
            default=False,
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

    if args.verbose:
        print('Preparing data ...')
    SYS = system.Supercell(args.TRAJECTORY,
                           fmt=args.T_format,
                           range=args.range,
                           units=args.T_units,
                           fn_topo=args.TOPOLOGY,
                           # --- generate mol centers, costly
                           wrap_molecules=False,
                           )
    MOMENTS = trajectory.MOMENTS(args.MOMENTS,
                                 fmt=args.M_format,
                                 range=args.range,
                                 units=args.M_units,
                                 )
    if args.electronic_centers is not None:
        CENTERS = trajectory.XYZ(args.electronic_centers,
                                 fmt=args.EC_format,
                                 range=args.range,
                                 units=args.EC_units,
                                 # fn_topo=args.TOPOLOGY,
                                 )
    if args.verbose:
        print('')

    def _get_batch(batch=None):
        _return = (
                MOMENTS.expand(batch=batch, ignore_warning=True),
                SYS.XYZ.expand(batch=batch, ignore_warning=True),
                )
        if args.electronic_centers is not None:
            _return += (CENTERS.expand(batch=batch, ignore_warning=True),)
        else:
            _return += (None,)
        return _return

    _iframe = 0
    while True:
        if args.verbose:
            print(f'Loading batch [{_iframe}:{_iframe+args.batch_size}]')
        ELE, NUC, WC = _get_batch(batch=args.batch_size)
        if None in [ELE, NUC]:
            if args.verbose:
                print('--- END OF TRAJECTORY')
            break
        if not args.do_not_join:
            if args.verbose:
                print('Wrapping molecules ...')
            NUC.wrap_molecules(SYS.mol_map)
        else:
            if args.verbose:
                print('Computing molecular centers ...')
            NUC.get_center_of_mass(mask=SYS.mol_map, join_molecules=False)

        if args.verbose:
            print('Assembling moments ...')
        #     _trajectory += (WC,)

        # --- test for neutrality of charge
        if (_total_charge := ELE.n_atoms * (-2) +
                constants.symbols_to_valence_charges(NUC.symbols).sum()) != 0.:
            warnings.warn(f'Got non-zero cell charge {_total_charge}!',
                          config.ChirPyWarning, stacklevel=2)

        n_map = np.array(SYS.mol_map)
        _cell = SYS.cell_aa_deg

        # for _iframe, (_frame) in enumerate(zip(*_trajectory)):

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

        # --- ensure that electronic centers have the same order
        #     (NB: not guaranteed by CPMD output)
        #     + assignment
        #     This assumes that the number of centers per nucleus
        #     does not change.
        # ToDo: use cython
        # wc_reference = []
        wc_origins_aa = []

        import copy
        for _iiframe in tqdm.tqdm(
                              range(ELE.n_frames),
                              disable=not args.verbose,
                              desc='map electronic centers --> nuclei',
                              ):
            # --- find nearest nucleus and add electron pair to it
            N = mapping.nearest_neighbour(
                    gauge_e.r_au[_iiframe] * constants.l_au2aa,
                    gauge_n.r_au[_iiframe] * constants.l_au2aa,
                    cell=_cell
                    )
            # _slist = np.argsort(N)
            _e_map = n_map[N]
            _slist = np.argsort(_e_map)

            # --- tweak OriginGauge
            gauge_e.r_au[_iiframe] = copy.deepcopy(gauge_e.r_au[_iiframe, _slist])
            gauge_e.c_au[_iiframe] = copy.deepcopy(gauge_e.c_au[_iiframe,_slist])
            gauge_e.m_au[_iiframe] = copy.deepcopy(gauge_e.m_au[_iiframe,_slist])
            # --- d_au and q_au of Wannier centers do not have to be sorted for
            # mol gauge
            gauge_e.q_au[_iiframe] = copy.deepcopy(gauge_e.q_au[_iiframe, _slist])
            if args.position_form:
                gauge_e.d_au[_iiframe] = copy.deepcopy(gauge_e.d_au[_iiframe,_slist])

            wc_origins_aa.append(copy.deepcopy(NUC.pos_aa[_iiframe, N][_slist]))
            # wc_reference.append(N)

        # N_map = np.sort(N)
        # e_map = n_map[N_map]
        e_map = np.sort(_e_map)
        # print(n_map)
        # print(tuple(enumerate(N)))
        # print(N_map)
        # print(e_map)
        wc_origins_aa = np.array(wc_origins_aa)
        # wc_reference = np.array(wc_reference)
        gauge_e.shift_origin_gauge(wc_origins_aa)
        # for _inuc in range(gauge_n.n_units):
        #     _ind = np.nonzero(wc_reference == _inuc)
        #     print(gauge_e.c_au[_ind].shape)
        #     gauge_n.c_au[_ind[0], _inuc] += gauge_e.c_au[_ind[0], _ind[1]]
        #     gauge_n.m_au[_ind[0], _inuc] += gauge_e.m_au[_ind[0], _ind[1]]
        #     gauge_n.q_au[_ind[0], _inuc] += gauge_e.q_au[_ind[0], _ind[1]]
        #     if args.position_form:
        #         gauge_n.d_au += gauge_e.d_au

        # --- combine nuclear and electronic contributions
        gauge = gauge_e + gauge_n
        # --- add frame to n_map
        assignment = np.concatenate((e_map, n_map))

        # --- shift to molecular origins
        _com = NUC.mol_com_aa
        gauge.shift_origin_gauge(_com, assignment)

        # --- test for neutrality of charge
        if np.any((_mol := gauge.q_au != 0.0)):
            warnings.warn('Got non-zero charge for (frame, molecule) '
                          f'{tuple(zip(*np.where(_mol)))}: {gauge.q_au[_mol]}',
                          config.ChirPyWarning, stacklevel=2)

        # --- write output
        append = False
        if _iframe > 0:
            append = True

        if args.position_form:
            _data = np.concatenate((gauge.r_au*constants.l_au2aa,
                                    gauge.c_au,
                                    gauge.m_au,
                                    gauge.d_au), axis=-1)
        else:
            _data = np.concatenate((gauge.r_au*constants.l_au2aa,
                                    gauge.c_au,
                                    gauge.m_au), axis=-1)

        if args.verbose:
            print('Writing output ...')
        cpmd.cpmdWriter(
                 args.f,
                 _data,
                 frames=list(range(_iframe, _iframe+args.batch_size)),
                 append=append,
                 write_atoms=False)

        _iframe += args.batch_size

    if args.verbose:
        print('Done.')


if __name__ == "__main__":
    main()
