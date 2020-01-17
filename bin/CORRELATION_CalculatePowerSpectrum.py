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
from matplotlib import pyplot as plt

from chirpy.classes import system
from chirpy.physics import spectroscopy, constants


def main():
    '''Calculate and plot the power spectrum of the given trajectory through
       time-correlation of the atomic velocities..'''
    parser = argparse.ArgumentParser(
            description="Calculate and plot the power spectrum of the given "
            "trajectory through time-correlation of the atomic velocities.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="Input file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument(
            "--fn_vel",
            help="External trajectory file with velocities (optional). "
                 "Less efficient. Assumes atomic units. BETA",
            default=None,
            )
    parser.add_argument(
            "--extract_molecules",
            nargs='+',
            help="Write only coordinates of given molecular ids starting from 0 \
                    (requires a topology file).",
            default=None,
            type=int,
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
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
            "--fn_topo",
            help="Topology file containing metadata (cell, \
                    molecules, ...).",
            default=None,
            )
    parser.add_argument(
            "--subset",
            nargs='+',
            help="Use only a subset of atoms. Expects list of indices "
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
            "--filter_strength",
            help="Strength of signal filter (welch) for TCF pre-processing."
                 "Give -1 to remove the implicit size-dependent triangular "
                 "filter",
            default=0,
            type=int,
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
            "-f",
            help="Output file name",
            default='power.dat'
            )
    args = parser.parse_args()
    if args.subset is None:
        args.subset = slice(None)

    if args.fn_topo is None:
        del args.fn_topo

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)
    if args.range is None:
        args.range = (0, 1, float('inf'))

    largs = vars(args)
    _load = system.Supercell(args.fn, **largs)

    if args.fn_vel is not None:
        # --- A little inefficient to load args.fn first as it will no
        #     longer be used (but pos and vel should no be in extra files
        #     anyway)
        nargs = {}
        for _a in [
            'range',
            'fn_topo',
            'sort',
                   ]:
            nargs[_a] = largs.get(_a)

        _load_vel = system.Supercell(args.fn_vel, **nargs)
        _load.XYZ.merge(_load_vel.XYZ, axis=-1)

    extract_molecules = largs.pop('extract_molecules')

    if extract_molecules is not None:
        _load.extract_molecules(extract_molecules)

    # --- expand iterator
    _vel = np.array([_load.XYZ.vel_au[args.subset] for _fr in _load.XYZ])
    _pow = spectroscopy.get_power_spectrum(
                                _vel,
                                ts=args.ts * constants.femto,
                                weights=_load.XYZ.masses_amu,
                                flt_pow=args.filter_strength,
                                return_tcf=args.return_tcf
                                )

    plt.plot(_pow['omega'] * constants.E_Hz2cm_1, _pow['power'])
    plt.xlim(*args.xrange)

    plt.xlabel(r'$\tilde\nu$ in cm$^{-1}$')
    plt.ylabel('Power in ...')
    plt.title('Power spectrum')
    plt.show()
    if args.save:
        np.savetxt(
             args.f,
             np.array((_pow['omega'] * constants.E_Hz2cm_1, _pow['power'])).T,
             header='omega in cm-1, power spectral density'
             )

    if args.return_tcf:
        plt.plot(np.arange(len(_pow['tcf_velocities'])) * args.ts / 1000,
                 _pow['tcf_velocities'])
        plt.xlabel(r'$\tau$ in ps')
        plt.ylabel('TCF in ...')
        plt.title('Time-correlation function of atomic velocities')
        plt.show()
        if args.save:
            np.savetxt(
                  'tcf_' + args.f,
                  np.array((
                      np.arange(len(_pow['tcf_velocities'])) * args.ts,
                      _pow['tcf_velocities']
                      )).T,
                  header='time in fs, time-correlation function of velocities'
                  )


if __name__ == "__main__":
    main()
