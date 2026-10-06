# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2024, The ChirPy Developers.
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
# ----------------------------------------------------------------------

import unittest
import os
from math import isclose
import numpy as np
import warnings

from chirpy import constants
from chirpy.config import ChirPyWarning
from chirpy.physics import statistical_mechanics, spectroscopy, \
    classical_electrodynamics
from chirpy.physics import kspace
from chirpy.classes import trajectory
# kspace, modern_theory_of_magnetisation

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_files'


class TestConstants(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_symbols_to_masses(self):
        self.assertListEqual(
                constants.symbols_to_masses(('C', 'H', 'D', 'P')).tolist(),
                [12.011, 1.008, 2.01410177784, 30.973761998],
                )

    def test_symbols_to_symbols_detect_element(self):
        # --- detect_element: guess the chemical element from an arbitrary
        # atom kind/name label (e.g. force-field style labels) by
        # progressively truncating from the right until a known element
        # symbol is matched; a warning is issued whenever guessing kicks in
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            detected = constants.symbols_to_symbols(
                    ('OW', 'HW1', 'NA+', 'C')
                    )
            msgs = [str(_w.message) for _w in w]

        self.assertTupleEqual(detected, ('O', 'H', 'Na', 'C'))
        # --- exact element symbols are recognised without any guessing
        self.assertFalse(any('C' == _m for _m in msgs))
        self.assertTrue(any('OW --> O' in _m for _m in msgs))
        self.assertTrue(any('HW1 --> H' in _m for _m in msgs))
        self.assertTrue(any('NA+ --> Na' in _m for _m in msgs))

        # --- unresolvable labels fall back to the original label itself
        # (fill_value='self')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            fallback = constants.symbols_to_symbols(('##!!',))
        self.assertTupleEqual(fallback, ('##!!',))

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

    def test_time_correlation_function(self):
        n = 200
        t = np.arange(n)
        signal = np.cos(2 * np.pi * t / 20.0)

        tcf = statistical_mechanics.time_correlation_function(
                signal, finite_size_correction=False)

        # --- shape: full convolution of two length-n signals
        self.assertEqual(tcf.shape, (2 * n - 1,))

        # --- zero-lag value equals the (biased) signal energy
        self.assertAlmostEqual(tcf[0], np.sum(signal**2), places=6)

        # --- symmetric about zero lag (tcf[+k] == tcf[-k])
        self.assertTrue(np.allclose(tcf[1:], tcf[1:][::-1], atol=1e-6))

        # --- explicit auto-correlation (1 arg) matches cross-correlation
        #     of the signal with itself (2 args)
        tcf_cross = statistical_mechanics.time_correlation_function(
                signal, signal, finite_size_correction=False)
        self.assertTrue(np.allclose(tcf, tcf_cross))

        # --- mismatched shapes raise
        with self.assertRaises(ValueError):
            statistical_mechanics.time_correlation_function(
                    signal, signal[:-1])

        # --- too many arguments raise
        with self.assertRaises(TypeError):
            statistical_mechanics.time_correlation_function(
                    signal, signal, signal)

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

        self.assertTrue(np.allclose(
            He,
            [0.0, 259.64351675199623, 861.543369563816,
             1419.6864797977644, 1631.9085469653362, 1455.5758919071563,
             1056.3539740488634, 639.7470438031493,
             328.23926631882813, 144.07447416251298],
            ))

        self.assertTrue(np.allclose(
            Ne,
            [0.00000000e+00, 2.28505493e+03, 3.56265819e+03, 1.66719300e+03,
             3.28930567e+02, 3.04352465e+01, 1.38485263e+00, 3.17814030e-02,
             3.73460406e-04, 2.26907524e-06,]
            ))
        self.assertTrue(np.allclose(
            Ar,
            [0.00000000e+00, 4.67919984e+03, 2.89848070e+03, 2.91239971e+02,
             6.66784843e+00, 3.86920937e-02, 5.96706032e-05, 2.50835460e-08,
             2.91787691e-12, 0.00000000e-00]
            ))

    def test_spectral_density(self):
        P = 100
        N = 1000
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

        omega, S, R = statistical_mechanics.spectral_density(
                                                sig,
                                                ts=1 / N * P,
                                                window_length=100,
                                                finite_size_correction=True,
                                                )

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
                                  mode='AB',
                                  window_length_au=len(_load.vel_au)*ts,
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

    def test_shift_magnetic_origin_gauge(self):
        _m = classical_electrodynamics.shift_magnetic_origin_gauge(
                np.array([1.2, 3, -1]),  # j
                np.array([0., 0., 0.]),
                np.array([-1., 3., 0.1]),  # r
                np.array([0., 0., 0.])
                )  # --> m
        # m = 0.5 * r × j
        # 0.5 * (-1., 3., 0.1) × (1.2, 3, -1) = (-1.65, -0.44, -3.3)
        self.assertListEqual(_m.tolist(), [-1.65, -0.44, -3.3])

        _m = classical_electrodynamics.shift_magnetic_origin_gauge(
                np.array([1.2, 3, -1]),
                np.array([1., 2., 0.]),
                np.array([-1., 0., 0.1]),
                np.array([-1., 1., -0.1])
                )
        self.assertListEqual(_m.tolist(), [1.2, 2.12, 0.6])

        # -- periodic
        _m = classical_electrodynamics.shift_magnetic_origin_gauge(
                np.array([1.2, 3, -1]),
                np.array([1., 2., 0.]),
                np.array([-1., 0., 0.1]),
                np.array([-1., 1., -0.1]),
                cell_au_deg=np.array([10., 0.7, 10., 90., 90., 90.])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [0.85, 2.12, 0.18])

        # -- one origin ---> multiple origins
        _m = classical_electrodynamics.shift_magnetic_origin_gauge(
                np.array([[1.2, 3, -1], [1.2, 3, -1]]),
                np.array([[1., 2., 0.], [1., 2., 0.]]),
                np.array([-1., 0., 0.1]),
                np.array([[-1., 1., -0.1], [-1., 3., -0.1]])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [[1.2, 2.12, 0.6], [2.2, 2.12, 1.8]])

        # -- multiple origins ---> one origin
        _m = classical_electrodynamics.shift_magnetic_origin_gauge(
                np.array([[1.2, 3, -1], [1.2, 3, -1]]),
                np.array([[1., 2., 0.], [1., 2., 0.]]),
                np.array([[-1., 0., 0.1], [-1., -2., 0.1]]),
                np.array([-1., 1., -0.1])
                )
        self.assertListEqual(np.round(_m, decimals=2).tolist(),
                             [[1.2, 2.12, 0.6], [2.2, 2.12, 1.8]])

    def test_shift_electric_origin_gauge(self):
        # mu(B) = mu(A) + q * (o_a - o_b)
        _mu = classical_electrodynamics.shift_electric_origin_gauge(
                np.array([2.0]),
                np.array([1.2, 3.0, -1.0]),
                np.array([-1., 3., 0.1]),
                np.array([0., 0., 0.]),
                )
        self.assertListEqual(np.round(_mu, decimals=6).tolist(),
                             [[-0.8, 9.0, -0.8]])

        # --- shifting there and back returns the original dipole moment
        _mu_back = classical_electrodynamics.shift_electric_origin_gauge(
                np.array([2.0]),
                _mu[0],
                np.array([0., 0., 0.]),
                np.array([-1., 3., 0.1]),
                )
        self.assertListEqual(np.round(_mu_back, decimals=6).tolist(),
                             [[1.2, 3.0, -1.0]])

        # -- periodic
        _mu_pbc = classical_electrodynamics.shift_electric_origin_gauge(
                np.array([2.0]),
                np.array([1.2, 3.0, -1.0]),
                np.array([-1., 3., 0.1]),
                np.array([0., 0., 0.]),
                cell_au_deg=np.array([10., 0.7, 10., 90., 90., 90.])
                )
        self.assertTupleEqual(_mu_pbc.shape, (1, 3))


class TestKspace(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_k_get_cell(self):
        R, K = kspace.k_get_cell(4, 4, 4, 8., 8., 8.)
        self.assertTupleEqual(R.shape, (4, 4, 4))
        self.assertTupleEqual(K.shape, (4, 4, 4))
        self.assertAlmostEqual(R.min(), 0.0, places=6)

        # -- refining the grid while keeping the same box: minimum
        # k-vector magnitude (k=0) stays 0, real-space spans the same box
        R2, K2 = kspace.k_get_cell(8, 8, 8, 8., 8., 8.)
        self.assertAlmostEqual(R2.max(), R.max(), places=6)

    def test_k_potential(self):
        n = 8
        cell_au = np.eye(3)
        rho = np.zeros((n, n, n))
        rho[n // 2, n // 2, n // 2] = 1.0

        R, V = kspace.k_potential(rho, cell_au)
        self.assertTupleEqual(V.shape, (n, n, n))

        # --- Coulomb potential of a point charge is maximal at the
        # charge position
        self.assertEqual(np.argmax(V), np.ravel_multi_index(
                (n // 2, n // 2, n // 2), V.shape))
