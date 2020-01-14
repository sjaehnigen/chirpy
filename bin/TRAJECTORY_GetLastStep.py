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

from chirpy.classes import system


def create_name(F):
    return F.split('.')[0] + '_last_step' + '.xyz'


def main():
    '''Extract last frame from a trajectory.'''
    parser = argparse.ArgumentParser(
            description="Extract last frame from a trajectory.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--cell_aa_deg",
                        nargs=6,
                        help="Orthorhombic cell parametres a b c al be ga in \
                                angstrom/degree",
                        default=[0.0, 0.0, 0.0, 90., 90., 90.]
                        )
    parser.add_argument("-f",
                        help="Output XYZ file name (auto: 'fn_[step].xyz')",
                        default=create_name
                        )

    args = parser.parse_args()
    args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)

    if args.f is create_name:
        args.f = create_name(args.fn)

    _load = system.Molecule(**vars(args)).XYZ
    _load._unwind()
    _load._frame.write(args.f)


if __name__ == "__main__":
    main()
