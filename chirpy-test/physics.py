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

import unittest
import os
from math import isclose
import numpy as np
import warnings

from chirpy.physics import constants, statistical_mechanics, spectroscopy, \
    classical_electrodynamics
from chirpy.classes import trajectory
# kspace, modern_theory_of_magnetisation

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestConstants(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_symbols_to_masses(self):
        self.assertListEqual(
                constants.symbols_to_masses(('C', 'H', 'D', 'P')).tolist(),
                [12.0107, 1.00794, 2.014101778, 30.973761],
                )

    def test_numbers_to_symbols(self):
        self.assertTupleEqual(
                constants.numbers_to_symbols([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F'),
                )

    def test_units(self):
        self.assertTrue(isclose(
                                4 * constants.pi
                                * constants.eps0_si * constants.hbar_si**2
                                / constants.e_si**2 / constants.m_e_si,
                                constants.a0_si,
                                rel_tol=1E-6
                                ))
        self.assertTrue(isclose(
                                constants.hbar_cgs**2
                                / constants.e_cgs**2 / constants.m_e_si
                                / constants.kilo,
                                constants.a0_cgs,
                                rel_tol=1E-6
                                ))
        self.assertTrue(isclose(
                                constants.e_si**4 * constants.m_e_si
                                / 4 / constants.eps0_si**2 / constants.h_si**2,
                                constants.E_au,
                                rel_tol=1E-6
                                ))


class TestStatisticalMechanics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_temperature_from_energies(self):
        E = statistical_mechanics.kinetic_energies([
                  [0.001, 0.0230, 0.000],
                  [0.0023, 0.00, 0.030]
                  ], [12.01, 15.99])
        self.assertListEqual(E.tolist(),
                             [5.801616036053368, 13.193690517437869])

    def test_maxwell_boltzmann_distribution(self):
        He = statistical_mechanics.maxwell_boltzmann_distribution(
                298.15,  4.00260, option='velocity')
        Ne = statistical_mechanics.maxwell_boltzmann_distribution(
                298.15, 20.17976, option='velocity')
        Ar = statistical_mechanics.maxwell_boltzmann_distribution(
                298.15, 39.95, option='velocity')

        vel_si = np.linspace(0, 2500, 10)
        vel_au = vel_si * constants.v_si2au

        He = list(map(He, vel_au))
        Ne = list(map(Ne, vel_au))
        Ar = list(map(Ar, vel_au))

        self.assertListEqual(
            He,
            [0.0, 259.64351675199623, 861.543369563816,
             1419.6864797977644, 1631.9085469653362, 1455.5758919071563,
             1056.3539740488634, 639.7470438031493,
             328.23926631882813, 144.07447416251298])
        self.assertListEqual(
            Ne,
            [0.0, 2285.0583431608047, 3562.659032223463,
             1667.1898975979307,
             328.9289905273825, 30.434985841657625, 1.384834386413647,
             0.031780811151197186,
             0.0003734511042990659, 2.269002561323683e-06])
        self.assertListEqual(
            Ar,
            [0.0, 4679.204911222703, 2898.4766305923536, 291.23835364271923,
             6.667772676402412, 0.03869136521866931, 5.9668935143188504e-05,
             2.5082574248733346e-08,
             2.917727567102299e-12, 9.484119560358289e-17])

    def test_spectral_density(self):
        P = 10000
        N = 100000
        X = np.linspace(0, P * 2*np.pi, N).reshape(N, 1)
        # freq = np.random.random(10) * np.pi
        freq = [
                2.28006930,
                0.90912777,
                2.06137591,
                2.93732800,
                0.16941237,
                0.30892809,
                1.48137192,
                3.07558759,
                2.19663137,
                2.46566211,
                0.91958214,
                0.10000100,
                ]
        sig = np.zeros_like(X)
        for f in freq:
            sig += np.sin(f * X)

        omega, S, R = statistical_mechanics.spectral_density(sig,
                                                             ts=1 / N * P,
                                                             flt_pow=-1)

        S /= np.amax(S)
        for _i in np.round(freq, decimals=2):
            self.assertIn(_i, np.unique(np.round(omega[S > 0.2], decimals=2)))


class TestSpectroscopy(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_power_from_tcf(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            _load = trajectory._XYZTrajectory.load(self.dir + '/ALANINE_NVT_3')
        # --- arbitrary ts value
        ts = 2
        POW = spectroscopy.power_from_tcf(
                                  _load.vel_au,
                                  ts_au=ts,
                                  weights=_load.masses_amu*constants.m_amu_au,
                                  average_atoms=False,
                                  cc_mode='A',
                                  flt_pow=-1,
                                  )

        self.assertAlmostEqual(
                np.mean(constants.k_B_au * 344 / (
                     POW['power'].sum(axis=1) * 2*np.pi * 2 * POW['freq'][1]
                     )),
                1.0,
                places=2
                )
        self.assertAlmostEqual(
                np.mean(constants.k_B_au * 344 / (
                     POW['power'].sum(axis=1) * 2*np.pi / _load.n_frames/ts
                     )),
                1.0,
                places=2
                )


class TestClassicalElectrodyanmics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_switch_magnetic_origin_gauge(self):
    def test_switch_magnetic_origin_gauge(self):
        _m = classical_electrodynamics.switch_magnetic_origin_gauge(
                np.array([1.2, 3, -1]),
                np.array([1., 2., 0.]),
                np.array([-1., 0., 0.1]),
                np.array([-1., 1., -0.1])
                )
        self.assertListEqual(_m.tolist(), [1.2, 2.12, 0.6])

        # -- periodic
        _m = classical_electrodynamics.switch_magnetic_origin_gauge(
                np.array([1.2, 3, -1]),
                np.array([1., 2., 0.]),
                np.array([-1., 0., 0.1]),
                np.array([-1., 1., -0.1]),
                cell_au_deg=np.array([10., 0.7, 10., 90., 90., 90.])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [0.85, 2.12, 0.18])

        # -- one origin ---> multiple origins
        _m = classical_electrodynamics.switch_magnetic_origin_gauge(
                np.array([[1.2, 3, -1], [1.2, 3, -1]]),
                np.array([[1., 2., 0.], [1., 2., 0.]]),
                np.array([-1., 0., 0.1]),
                np.array([[-1., 1., -0.1], [-1., 3., -0.1]])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [[1.2, 2.12, 0.6], [2.2, 2.12, 1.8]])

        # -- multiple origins ---> one origin
        _m = classical_electrodynamics.switch_magnetic_origin_gauge(
                np.array([[1.2, 3, -1], [1.2, 3, -1]]),
                np.array([[1., 2., 0.], [1., 2., 0.]]),
                np.array([[-1., 0., 0.1], [-1., -2., 0.1]]),
                np.array([-1., 1., -0.1])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [[1.2, 2.12, 0.6], [2.2, 2.12, 1.8]])
