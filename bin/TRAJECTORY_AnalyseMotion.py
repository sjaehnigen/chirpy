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

from chirpy.visualise import timeline
from chirpy.topology import motion
from chirpy.topology import mapping
from chirpy.classes import system


def main():
    parser = argparse.ArgumentParser(
         description="Analyse motion of atoms in trajectory and plot results.",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
         )
    parser.add_argument(
            "fn",
            help="Trajectory file (xyz, ...)"
            )
    parser.add_argument(
            "--fn_vel",
            help="Additional trajectory file with velocities (optional)."
                 "Assumes atomic units.",
            default=None,
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Range of frames to read (start, step, stop)",
            default=None,
            type=int,
            )
    # parser.add_argument("--fn_topo",
    #                     help="Topology file containing metadata (cell, \
    #                             molecules, ...).",
    #                     default=None,
    #                     )
    parser.add_argument(
            "--subset",
            nargs='+',
            help="Atom list (id starting from 0).",
            type=int,
            default=None,
            )
    parser.add_argument(
            "--noplot",
            default=False,
            action='store_true',
            help="Do not plot results."
            )
    args = parser.parse_args()
    no_plot = args.noplot
    if args.subset is None:
        del args.subset
    if args.range is None:
        del args.range

    if no_plot:
        plot = 0
    else:
        plot = 1

    largs = vars(args)
    _load = system.Supercell(args.fn, **largs)
    _w = _load.XYZ.masses_amu
    if args.fn_vel is not None:
        _load_vel = system.Supercell(args.fn_vel, **largs)

    def get_p_and_v():
        # --- old but working: could use iterator merge
        try:
            while True:
                next(_load.XYZ)
                _p = _load.XYZ.pos_aa
                _v = _load.XYZ.vel_au
                if args.fn_vel is not None:
                    next(_load_vel.XYZ)
                    if bool(_load_vel.XYZ._is_similar(_load.XYZ)):
                        _v = _load_vel.XYZ.pos_aa
                yield _p, _v

        except StopIteration:
            pass

    def get_results():
        _it = get_p_and_v()
        try:
            while True:
                _p, _v = next(_it)
                _com = mapping.cowt(_p, _w, **largs)
                _lin = motion.linear_momenta(_v, _w, **largs)
                _ang = motion.angular_momenta(_p, _v, _w, origin=_com, **largs)
                yield _com, _lin, _ang

        except StopIteration:
            pass

    center_of_masses, linear_momenta, angular_momenta = np.array(
            tuple(zip(*[_r for _r in get_results()]))
            )
    step_n = np.arange(len(center_of_masses))

    timeline.show_and_interpolate_array(
         step_n, center_of_masses[:, 0], 'com_x', 'step', 'com_x', plot)
    timeline.show_and_interpolate_array(
         step_n, center_of_masses[:, 1], 'com_y', 'step', 'com_y', plot)
    timeline.show_and_interpolate_array(
         step_n, center_of_masses[:, 2], 'com_z', 'step', 'com_z', plot)
    timeline.show_and_interpolate_array(
         step_n, linear_momenta[:, 0], 'lin_mom_x', 'step', 'lin_mom_x', plot)
    timeline.show_and_interpolate_array(
         step_n, linear_momenta[:, 1], 'lin_mom_y', 'step', 'lin_mom_y', plot)
    timeline.show_and_interpolate_array(
         step_n, linear_momenta[:, 2], 'lin_mom_z', 'step', 'lin_mom_z', plot)
    timeline.show_and_interpolate_array(
         step_n, angular_momenta[:, 0], 'ang_mom_x', 'step', 'ang_mom_x', plot)
    timeline.show_and_interpolate_array(
         step_n, angular_momenta[:, 1], 'ang_mom_y', 'step', 'ang_mom_y', plot)
    timeline.show_and_interpolate_array(
         step_n, angular_momenta[:, 2], 'ang_mom_z', 'step', 'ang_mom_z', plot)


if __name__ == "__main__":
    main()
