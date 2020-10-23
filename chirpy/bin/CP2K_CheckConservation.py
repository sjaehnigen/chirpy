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

import sys
import argparse
import numpy as np
import warnings

from chirpy.interface import cp2k
from chirpy.visualise import timeline


def main(*args):

    if len(args) == 0:
        parser = argparse.ArgumentParser(
                description="check_convergence",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
                )
        parser.add_argument(
                "fn",
                help="CP2K energy file (*.ener)"
                )
        parser.add_argument(
                "--verbose",
                default=False,
                action='store_true',
                help="verbose output"
                )
        parser.add_argument(
                "--noplot",
                default=False,
                action='store_true',
                help="plot output"
                )
        args = parser.parse_args()
        fn = args.fn
        verbose = args.verbose
        no_plot = args.noplot
    else:
        fn = tuple(args)
        verbose = False
    if verbose:
        print(' '.join(sys.argv))
        print('fn_ener: %s' % fn)
    if no_plot:
        plot = 0
    else:
        plot = 1

    step_n, time, temp, kin, pot, cqty = cp2k.read_ener_file(fn)

    if verbose:
        print(' '.join(sys.argv))
        print('kinetic energy ', kin)
        print('potential energy ', pot)
        print('temperature ', temp)
        print('conserved quantity ', cqty)

    delta_t = np.diff(time)
    if not len(np.unique(delta_t)) == 1:
        warnings.warn("CRITICAL: Found varying timesteps!", stacklevel=2)
    timeline.show_and_interpolate_array(
            step_n[1:], np.diff(time), 'timestep', 'step', 'time in fs', plot)
    timeline.show_and_interpolate_array(
            step_n, time, 'time', 'step', 'time in fs', plot)
    timeline.show_and_interpolate_array(
            step_n, temp, 'temperature', 'step', 'T', plot)
    timeline.show_and_interpolate_array(
            step_n, kin, 'kinetic energy', 'step', 'Ekin', plot)
    timeline.show_and_interpolate_array(
            step_n, pot, 'potential energy', 'step', 'Epot', plot)
    timeline.show_and_interpolate_array(
            step_n, cqty, 'conserved quantity', 'step', 'C. Qty', plot)


if __name__ == "__main__":
    main()
