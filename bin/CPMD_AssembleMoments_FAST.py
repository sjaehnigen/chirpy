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

from chirpy.classes import system, trajectory
from chirpy.topology import mapping as mp
from chirpy.physics import constants
from chirpy.physics import classical_electrodynamics as ed
from chirpy.interface import cpmd


def main():
    parser = argparse.ArgumentParser(
            description="Process CPMD MOMENTS output of electronic (Wannier)\
                         states and add (classical) nuclear contributions to\
                         generate molecular moments based on a given\
                         topology.\
                         FAST: The entire data is loaded into memory before\
                         being processed.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "TOPOLOGY",
            help="pdb file with topology including cell specifications"
            )
    parser.add_argument(
            "TRAJECTORY",
            help="TRAJECTORY file from CPMD"
            )
    parser.add_argument(
            "MOMENTS",
            help="MOMENTS file from CPMD"
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Frame range for reading CPMD files.",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--hag",
            action='store_true',
            default=False,
            help="Use heavy atom gauge (BETA; work in progress)."
            )
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
                             wrap_mols=False,
                             )

    _moms_e = trajectory.MOMENTS(args.MOMENTS,
                                 fmt='cpmd',
                                 range=args.range
                                 )

    n_map = np.array(_traj.mol_map)
    n_mols = len(set(_traj.mol_map))

    # --- load data into memory
    _traj = _traj.XYZ
    e_data = np.array([_moms_e.data for _m_fr in _moms_e])
    _traj = _traj._to_trajectory()
    # --- wrap molecules for all frames at once
    _traj.wrap_molecules(n_map)
    n_data = _traj.data

    ZV = np.array(constants.symbols_to_valence_charges(_traj.symbols))

    _cell = _traj.cell_aa_deg
    _cell[:3] *= constants.l_aa2au

    print('Finished loading data.')
    print('Start assembling moments...')
    for _iframe, (_n_data, _e_data) in enumerate(zip(n_data, e_data)):
        #
        #

        _r_n = _n_data[:, :3] * constants.l_aa2au
        _c_n = ed.current_dipole_moment(_n_data[:, 3:6], ZV)
        _m_n = np.zeros_like(_c_n)

        _r_e = _e_data[:, :3]
        _c_e = _e_data[:, 3:6]
        _m_e = _e_data[:, 6:9]

        # --- assign Wannier centers to atoms
        _dists = mp.distance_matrix(_r_e, _r_n, cell=_cell)
        if args.hag:
            for _ia, _s in enumerate(_traj.symbols):
                if _s == 'H':
                    _dists[:, _ia] += 1E10

        e_map = n_map[np.argmin(_dists, axis=1)]

        # --- decompose data into molecular contributions
        mol_com = _traj.mol_com_aa[_iframe] * constants.l_aa2au
        _r_n, _c_n, _m_n = map(lambda x: mp.dec(x, n_map), [_r_n, _c_n, _m_n])
        _r_e, _c_e, _m_e = map(lambda x: mp.dec(x, e_map, n_ind=n_mols),
                               [_r_e, _c_e, _m_e])

        # --- translate Wannier-centre/nuclear gauge to molecular com
        _m_e = [ed.switch_origin_gauge(_c, _m, _r, _o, cell_au_deg=_cell)
                for _o, _r, _c, _m in zip(mol_com, _r_e, _c_e, _m_e)]
        _m_n = [ed.switch_origin_gauge(_c, _m, _r, _o, cell_au_deg=_cell)
                for _o, _r, _c, _m in zip(mol_com, _r_n, _c_n, _m_n)]

        # --- calculate the molecular current dipole moment
        # ToDo: old code
        # if args.hag:
        #    pass
        #     mol_c = np.array([_c_n[_i].sum(axis=0) for _i in range(n_mols)])
        #     mol_m = np.zeros(mol_c.shape)
        #     for _i in range(len(_c_e)):
        #         mol_c[_i] += _c_e[_i].sum(axis=0)
        #         mol_m[_i] += _m_e[_i].sum(axis=0)

        # else:
        mol_c = np.array([_c_n[_i].sum(axis=0) + _c_e[_i].sum(axis=0)
                          for _i in range(n_mols)])
        # --- nuclear contribution, _m_n, zero in classical limit
        mol_m = np.array([_m_n[_i].sum(axis=0) + _m_e[_i].sum(axis=0)
                          for _i in range(n_mols)])

        append = False
        if _iframe > 0:
            append = True

        cpmd.cpmdWriter(
                  args.f,
                  np.array([np.concatenate((mol_com, mol_c, mol_m), axis=-1)]),
                  frame=_iframe,
                  append=append,
                  write_atoms=False)
    print('Done')


if __name__ == "__main__":
    main()