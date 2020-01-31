#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------

import unittest
from math import isclose
import numpy as np

from ..physics import constants, statistical_mechanics
# classical_electrodynamics
# kspace, modern_theory_of_magnetisation, spectroscopy


class TestConstants(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_symbols_to_masses(self):
        self.assertListEqual(
                constants.symbols_to_masses(('C', 'H', 'D', 'P')).tolist(),
                [12.01115, 1.00797, 2.01410, 30.97376],
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
    # --- insufficiently tested

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
        freq = np.random.random(10) * np.pi
        sig = np.zeros_like(X)
        for f in freq:
            sig += np.sin(f * X)

        omega, S, R = statistical_mechanics.spectral_density(sig,
                                                             ts=1 / N * P,
                                                             flt_pow=-1)

        S /= np.amax(S)
        for _i in np.round(freq, decimals=2):
            self.assertIn(_i, np.unique(np.round(omega[S > 0.2], decimals=2)))
