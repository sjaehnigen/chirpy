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
import warnings
import numpy as np
from matplotlib import pyplot as plt

from chirpy.classes import trajectory
from chirpy.physics import spectroscopy, constants


def main():
    '''Calculate and plot vibrational spectra of the given trajectory through
       time-correlation of dipole moments.'''
    parser = argparse.ArgumentParser(
            description="Calculate and plot vibrational spectra of the given "
            "trajectory through time-correlation of dipole moments.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input MOMENTS file (in CPMD format)"
            )
    parser.add_argument(
            "--input_format",
            help="Input file format (e.g. cpmd; optional).",
            default='cpmd',
            )

    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
            default=None,
            type=float,
            )
    parser.add_argument(
            "--va",
            action='store_true',
            help="Calculate vibrational absorption (IR) spectrum.",
            default=False,
            )
    parser.add_argument(
            "--vcd",
            action='store_true',
            help="Calculate vibrational circular dichroism spectrum.",
            default=False,
            )
    parser.add_argument(
            "--range",
            nargs=3,
            help="Range of frames to read (start, step, stop)",
            default=None,
            type=int,
            )
    # parser.add_argument(
    #         "--kinds",
    #         nargs='+',
    #         help="List of kinds per frame.",
    #         default=[1],
    #         )
    parser.add_argument(
            "--subset",
            nargs='+',
            help="Use only a subset of kinds. Expects ist of indices "
                 "(id starting from 0).",
            type=int,
            default=None,
            )
    parser.add_argument(
            "--ts",
            help="Timestep in fs between used frames (according to given "
                 "--range!).",
            default=0.5,
            type=float,
            )
    parser.add_argument(
            "--origin_id",
            nargs='+',
            help="IDs of kinds whose positions to be sampled in the "
                 "distributed gauge (default: every particle in MOMENTS) "
                 "(id starting from 0).",
            type=int,
            default=None,
            )
    parser.add_argument(
            "--cutoff",
            help="Cutoff in angstrom to scale neighbouring moments "
                 "surrounding each molecular origin.",
            default=0.,
            type=float,
            )
    parser.add_argument(
            "--filter_strength",
            help="Strength of signal filter (welch) for TCF pre-processing."
                 "Give <0 to remove the implicit size-dependent triangular "
                 "filter",
            default=-1,
            type=float,
            )
    parser.add_argument(
            "--return_tcf",
            help="Return also the time-correlation function.",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--save",
            help="Save results to file.",
            action='store_true',
            default=False,
            )
    parser.add_argument(
            "--xrange",
            nargs=2,
            help="Plotted range of frequencies (in 1/cm; "
                 "does not effect --save).",
            default=[2000., 500.],
            type=float,
            )
    parser.add_argument(
            "--noplot",
            default=False,
            action='store_true',
            help="Do not plot results."
            )

    parser.add_argument(
            "-f",
            help="Output file name",
            default='spectrum.dat'
            )
    parser.add_argument(
            "--verbose",
            action='store_true',
            help="Print info and progress.",
            default=False
            )
    args = parser.parse_args()
    if args.subset is None:
        args.subset = slice(None)

    if args.origin_id is None:
        args.origin_id = slice(None)

    if args.range is None:
        args.range = (0, 1, float('inf'))

    largs = vars(args)
    largs.update({'fmt': args.input_format})
    _load = trajectory.MOMENTS(largs.pop('fn'), **largs)

    if not any([args.va, args.vcd]):
        warnings.warn("Neither --va nor --vcd argument set! Did nothing.",
                      RuntimeWarning, stacklevel=2)
        sys.exit(0)

    # --- expand iterator
    _p, _c, _m = np.split(
                      np.array([_load.data[args.subset] for _fr in _load]),
                      3,
                      axis=-1
                      )

    if args.cell_aa_deg is not None:
        _cell = np.array(args.cell_aa_deg)
        _cell[:3] *= constants.l_aa2au
    else:
        _cell = None

    _voa = {}
    _voa['va'] = []
    _voa['vcd'] = []
    _voa['tcf_va'] = []
    _voa['tcf_vcd'] = []
    # --- ToDo: differ mode according to args
    origins = _p.swapaxes(0, 1)[args.origin_id]
    for origin in origins:
        _tmp = spectroscopy._spectrum_from_tcf(
                                    _c, _m,
                                    positions_au=_p*constants.l_aa2au,
                                    mode='abs_cd',
                                    ts_au=args.ts * constants.t_fs2au,
                                    flt_pow=args.filter_strength,
                                    # --- example
                                    origin_au=origin*constants.l_aa2au,
                                    cutoff_au=args.cutoff*constants.l_aa2au,
                                    cell_au_deg=_cell
                                    )

        _voa['va'].append(_tmp['abs'])
        _voa['vcd'].append(_tmp['cd'])
        _voa['tcf_va'].append(_tmp['tcf_abs'])
        _voa['tcf_vcd'].append(_tmp['tcf_cd'])

    _voa['freq'] = _tmp['freq']
    _voa['va'] = np.array(_voa['va']).sum(axis=0) / len(origins)
    _voa['vcd'] = np.array(_voa['vcd']).sum(axis=0) / len(origins)
    _voa['tcf_va'] = np.array(_voa['tcf_va']).sum(axis=0) / len(origins)
    _voa['tcf_vcd'] = np.array(_voa['tcf_vcd']).sum(axis=0) / len(origins)

    # --- plot
    labels = {
       'va': ('Vibrational absorption spectrum',
              'A in L / (cm · mol)'),
       'vcd': ('Vibrational circular dichroism spectrum',
               r'$\Delta$A in L / (cm · mol)'),
    }
    for _i in filter(args.__dict__.get, ['va', 'vcd']):
        if not args.noplot:
            plt.plot(_voa['freq'] * constants.E_aufreq2cm_1,
                     _voa[_i] * constants.Abs_au2L_per_cm_mol)
            plt.xlim(*args.xrange)
            plt.xlabel(r'$\tilde\nu$ in cm$^{-1}$')
            plt.title(labels[_i][0])
            plt.ylabel(labels[_i][1])
            plt.show()

        if args.save:
            np.savetxt(
                _i + '_' + args.f,
                np.array((_voa['freq'] * constants.E_aufreq2cm_1,
                          _voa[_i] * constants.Abs_au2L_per_cm_mol)).T,
                header='omega in cm-1, %s density' % _i
                )

        if args.return_tcf:
            if not args.noplot:
                plt.plot(np.arange(len(_voa['tcf_' + _i])) * args.ts / 1000,
                         _voa['tcf_' + _i])
                plt.xlabel(r'$\tau$ in ps')
                plt.ylabel('TCF in ...')
                plt.title('Time-correlation function for ' + _i)
                plt.show()
            if args.save:
                np.savetxt(
                     'tcf_' + _i + '_' + args.f,
                     np.array((
                         np.arange(len(_voa['tcf_' + _i])) * args.ts,
                         _voa['tcf_' + _i]
                         )).T,
                     header='time in fs, time-correlation function for ' + _i
                     )


if __name__ == "__main__":
    main()
