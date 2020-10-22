#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------

import argparse
import numpy as np

from chirpy.create.moments import OriginGauge
from chirpy.classes import system, trajectory
from chirpy.topology import mapping
from chirpy.physics import constants
from chirpy.interface import cpmd


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
#             help="Use heavy atom gauge (BETA; work in progress)."
#             )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='MOL'
            )

    args = parser.parse_args()
    if args.range is None:
        args.range = (0, 1, float('inf'))

    _traj = system.Supercell(args.TRAJECTORY,
                             fmt='cpmd',
                             range=args.range,
                             fn_topo=args.TOPOLOGY,
                             # --- this is costly depending on no of mols
                             wrap_molecules=True,
                             )

    _moms_e = trajectory.MOMENTS(args.MOMENTS,
                                 fmt='cpmd',
                                 range=args.range
                                 )

    n_map = np.array(_traj.mol_map)

    _cell = _traj.cell_aa_deg
    _cell[:3] *= constants.l_aa2au

    for _iframe, (_p_fr, _m_fr) in enumerate(zip(_traj.XYZ, _moms_e)):
        # --- generate classical nuclear moments
        gauge_n = OriginGauge(
               trajectory.MOMENTSFrame.from_classical_nuclei(_traj.XYZ._frame),
               cell=_cell
               )

        # --- load Wannier data and get nearest atom assignment
        gauge_e = OriginGauge(_moms_e, cell=_cell)
        e_map = n_map[mapping.nearest_neighbour(gauge_e.r, gauge_n.r,
                                                cell=_cell)]

        # --- combine nuclear and electronic contributions
        gauge = gauge_e + gauge_n
        assignment = np.concatenate((e_map, n_map))

        # --- switch to molecular origins
        _com = _traj.XYZ.mol_com_aa * constants.l_aa2au
        gauge.switch_origin_gauge(_com, assignment)

        # --- write output
        append = False
        if _iframe > 0:
            append = True

        cpmd.cpmdWriter(
                 args.f,
                 np.array([np.concatenate((_com, gauge.c, gauge.m), axis=-1)]),
                 frame=_iframe,
                 append=append,
                 write_atoms=False)


if __name__ == "__main__":
    main()
