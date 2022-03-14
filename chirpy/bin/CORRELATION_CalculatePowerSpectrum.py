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

import argparse
import numpy as np
from matplotlib import pyplot as plt

from chirpy.classes import system
from chirpy.physics import spectroscopy
from chirpy import config, constants


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
                 "Less efficient. Assumes atomic units.",
            default=None,
            )
    parser.add_argument(
            "--extract_molecules",
            nargs='+',
            help="Consider only coordinates of given molecular ids starting from 0 \
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
                 "(id starting from 0). If extract_molecules is set, it "
                 "applies after.",
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
            default='power.dat'
            )
    parser.add_argument(
            "--verbose",
            action='store_true',
            help="Print info and progress.",
            default=False,
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
    config.set_verbose(args.verbose)

    largs = vars(args)
    _files = [largs.pop('fn')]
    if args.fn_vel is not None:
        _files.append(args.fn_vel)
    _load = system.Supercell(*_files, **largs)

    extract_molecules = largs.pop('extract_molecules')

    if extract_molecules is not None:
        _load.extract_molecules(extract_molecules)

    # --- expand iterator
    _vel = np.array([_load.XYZ.vel_au[args.subset] for _fr in _load.XYZ])
    _pow = spectroscopy.power_from_tcf(
                                _vel,
                                ts_au=args.ts * constants.t_fs2au,
                                weights=_load.XYZ.masses_amu[args.subset],
                                flt_pow=args.filter_strength,
                                )

    # --- plot
    _POW_au2kJpermol = constants.E_au2J * constants.avog / constants.kilo
    if not args.noplot:
        plt.plot(_pow['freq'] * constants.E_aufreq2cm_1,
                 _pow['power'] * _POW_au2kJpermol)
        plt.xlim(*args.xrange)

        plt.xlabel(r'$\tilde\nu$ in cm$^{-1}$')
        plt.ylabel('Power in kJ / mol')
        plt.title('Power spectrum')
        plt.show()
    if args.save:
        np.savetxt(
           args.f,
           np.array((_pow['freq'] * constants.E_aufreq2cm_1,
                     _pow['power'] * _POW_au2kJpermol)).T,
           header='omega in cm-1, power spectral density'
           )

    if args.return_tcf:
        if not args.noplot:
            plt.plot(np.arange(len(_pow['tcf_power'])) * args.ts / 1000,
                     _pow['tcf_power'])
            plt.xlabel(r'$\tau$ in ps')
            plt.ylabel('TCF in ...')
            plt.title('Time-correlation function of atomic velocities')
            plt.show()

        if args.save:
            np.savetxt(
                  'tcf_' + args.f,
                  np.array((
                      np.arange(len(_pow['tcf_power'])) * args.ts,
                      _pow['tcf_power']
                      )).T,
                  header='time in fs, time-correlation function of velocities'
                  )


if __name__ == "__main__":
    main()
