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

import unittest
import os
import numpy as np
import warnings

from ..interface import cpmd

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestCPMD(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cpmdReader(self):
        for _i, _n in zip(['GEOMETRY', 'MOMENTS', 'TRAJECTORY'],
                          [(1, 208, 6), (5, 288, 9), (6, 208, 6)]):

            data = cpmd.cpmdReader(self.dir + '/' + _i,
                                   filetype=_i,
                                   symbols=['X']*_n[1])['data']

            self.assertTrue(np.array_equal(
                data,
                np.genfromtxt(self.dir + '/data_' + _i).reshape(_n)
                ))

        # Some Negatives
        with self.assertRaises(ValueError):
            data = cpmd.cpmdReader(self.dir + '/MOMENTS_broken',
                                   filetype='MOMENTS',
                                   symbols=['X']*288)['data']
            data = cpmd.cpmdReader(self.dir + '/MOMENTS',
                                   filetype='MOMENTS',
                                   symbols=['X']*286)['data']
        # Test range
        data = cpmd.cpmdReader(self.dir + '/' + _i,
                               filetype='TRAJECTORY',
                               symbols=['X']*_n[1],
                               range=(2, 3, 8),
                               )['data']
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_TRAJECTORY').reshape(_n)[2:8:3]
            ))

    def test_cpmdWriter(self):
        data = cpmd.cpmdReader(self.dir + '/TRAJECTORY',
                               filetype='TRAJECTORY',
                               symbols=['X']*208)['data']

        cpmd.cpmdWriter(self.dir + '/OUT', data, write_atoms=False)
        with self.assertRaises(ValueError):
            # --- sorted data
            cpmd.cpmdWriter(self.dir + '/OUT', data, symbols=['X', 'Y']*104, write_atoms=True)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data2 = cpmd.cpmdReader(self.dir + '/OUT',
                                    filetype='TRAJECTORY',
                                    symbols=cpmd.cpmd_kinds_from_file(self.dir
                                                                      + '/OUT')
                                    )['data']
        self.assertTrue(np.array_equal(data, data2))
        os.remove(self.dir + "/OUT")
