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


import sys
import argparse
import numpy as np
from chirpy.visualise import timeline


def main():
    sys.path = sys.path[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "fn",
            nargs=1,
            help="CP2K log file containing particle temperatures")
    args = parser.parse_args()
    fn = args.fn[0]

    with open(fn, 'r') as f:
        particles = np.array([line.strip().split()[1:]
                              for line in f
                              if "#" in line
                              and "TEMPERATURE REGIONS" not in line][3:])
    with open(fn, 'r') as f:
        step_time_temp = np.array(list(line.strip().split()
                                       for line in f
                                       if "#" not in line)[1:])

    data = np.append(step_time_temp, particles, axis=1).astype(float)

    timeline.show_and_interpolate_array(
            data[:, 0], data[:, 2], 'total system', 'step', 'T in K', 1)

    for _iy, _y in enumerate(data[:, 3:].T):
        timeline.show_and_interpolate_array(
            data[:, 0], _y, f'particle {_iy}',  'step', 'T in K', 1)


if __name__ == "__main__":
    main()
