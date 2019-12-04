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

import sys
import argparse
import numpy as np
from chirpy.interfaces import cp2k
from chirpy.visualisation import timeline


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
