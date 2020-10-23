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
from chirpy.classes import volume


def main():
    '''Write scalar volume data into file'''
    parser = argparse.ArgumentParser(
            description="Write volume data into file",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="File(s) that contain(s) volume data",
            nargs='+'
            )
    parser.add_argument(
            "--vector_field",
            help="FN contains vector field data.",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--sparsity",
            help="Reduce the meshsize by factor.",
            type=int,
            default=1
            )
    parser.add_argument(
            "--auto_crop",
            help="Crop grid edges.",
            action='store_true',
            default=False
            )
    parser.add_argument(
            "--crop_thresh",
            help="Lower threshold for cropping grid edges.",
            type=float,
            default=1.E-3
            )

    parser.add_argument(
            "-f",
            help="Output file name(s)",
            nargs='+',
            default=['out.cube'])
    args = parser.parse_args()

    if args.vector_field:
        system = volume.VectorField(*args.fn, **vars(args))
    else:
        system = volume.ScalarField(*args.fn, **vars(args))

    if args.auto_crop:
        system.auto_crop(thresh=args.crop_thresh)

    system.write(*args.f)


if(__name__ == "__main__"):
    main()
