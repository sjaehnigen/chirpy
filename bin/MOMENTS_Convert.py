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

from chirpy.classes import trajectory


def main():
    '''Convert and process trajectory'''
    parser = argparse.ArgumentParser(
            description="Convert and process trajectory",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input trajectory file."
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. xyz, pdb, cpmd; optional).",
            default=None,
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Range of frames to read (start, step, stop)",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--mask_frames",
            nargs='+',
            help="If True: Check for duplicate frames (based on comment line) "
                 "and skip them (may take some time).\n"
                 "If list of arguments: Skip given frames without further "
                 "checking (fast).",
            default=None
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
            default=None,
            )
    parser.add_argument(
            "-f",
            help="Output file name",
            default='out.cpmd'
            )
    parser.add_argument(
            "--output_format",
            help="Output file format (e.g. xyz, pdb, cpmd; optional).",
            default='cpmd',
            )
    args = parser.parse_args()

    # ToDo: MOMENTS requires atoimc units for cell (cell_au_deg)
    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    if args.range is None:
        args.range = (0, 1, float('inf'))

    i_fmt = args.input_format
    o_fmt = args.output_format
    if i_fmt is None:
        i_fmt = args.fn.split('.')[-1].lower()
    if o_fmt is None:
        o_fmt = args.f.split('.')[-1].lower()
    elif args.f == 'out.cpmd':
        args.f = 'out.' + o_fmt

    # --- Caution when passing all arguments to object!
    largs = vars(args)

    skip = largs.pop('mask_frames')
    if skip is not None:
        if skip[0] in ['True', 'False']:
            if skip[0] == 'True':
                skip = bool(skip[0])
                _load = trajectory.MOMENTS(args.fn, fmt=i_fmt, **largs)
                skip = _load.mask_duplicate_frames(verbose=False)
                largs.update({'skip': skip})
            else:
                _load = trajectory.MOMENTS(args.fn, fmt=i_fmt, **largs)
                largs.update({'skip': []})
        else:
            skip = [int(_a) for _a in skip]
            largs.update({'skip': skip})
            _load = trajectory.MOMENTS(args.fn, fmt=i_fmt, **largs)
    else:
        _load = trajectory.MOMENTS(args.fn, fmt=i_fmt, **largs)
        largs.update({'skip': []})

    if args.f not in ['None', 'False']:
        largs = {}
        if 'pp' in args:
            largs = {'pp': args.pp}
        _load.write(args.f, fmt=o_fmt, rewind=False, **largs)


if __name__ == "__main__":
    main()
