# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
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
import numpy as np
import warnings
import copy

from chirpy.read import modes as r_modes
from chirpy.read import coordinates as r_coordinates
from chirpy.read import grid as r_grid

from chirpy.write import modes as w_modes
from chirpy.write import coordinates as w_coordinates
from chirpy.write import grid as w_grid

from chirpy.config import ChirPyWarning
from chirpy import constants

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestModes(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_xvibsWriter(self):
        n_atoms, numbers, pos_aa, n_modes, freqs, modes = \
                r_modes.xvibsReader(self.dir + '/test.xvibs')
        w_modes.xvibsWriter(self.dir + '/out.xvibs',
                            n_atoms,
                            numbers,
                            pos_aa,
                            freqs,
                            modes)
        n_atoms2, numbers2, pos_aa2, n_modes2, freqs2, modes2 = \
            r_modes.xvibsReader(self.dir + '/out.xvibs')
        self.assertEqual(n_atoms2, n_atoms)
        self.assertEqual(n_modes2, n_modes)
        self.assertListEqual(numbers2, numbers)
        self.assertListEqual(freqs2, freqs)
        self.assertTrue(np.array_equal(pos_aa2, pos_aa))
        self.assertTrue(np.array_equal(modes2, modes))
        os.remove(self.dir + "/out.xvibs")


class TestCoordinates(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_xyzWriter(self):
        _ref = r_coordinates.xyzReader(self.dir + '/test_frame_posvel_pbc.xyz')
        data_r, symbols_r, comments_r = _ref
        data, symbols, comments = copy.deepcopy(_ref)
        w_coordinates.xyzWriter(self.dir + '/out.xyz', data, symbols,
                                comments=comments)
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')

        # --- units
        w_coordinates.xyzWriter(self.dir + '/out.xyz',
                                data,
                                symbols,
                                comments=comments,
                                units='default')
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')
        self.assertTrue(np.array_equal(data_r, data2))

        w_coordinates.xyzWriter(
                self.dir + '/out.xyz',
                data,
                symbols,
                comments=comments,
                units=3*[('length', 'aa')] + 3*[('velocity', 'au')]
                )
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')
        self.assertTrue(np.array_equal(data_r, data2))

        UNITS = 3*[('length', 'aa')] + 3*[('velocity', 'aa_ps')]
        w_coordinates.xyzWriter(
                self.dir + '/out.xyz',
                data,
                symbols,
                comments=comments,
                units=UNITS
                )

        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')
        data2[:, :, 3:] *= 1/constants.v_au2aa_fs/1000
        self.assertTrue(np.allclose(data_r, data2))

        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz', units=UNITS)
        self.assertTrue(np.allclose(data_r, data2))

        _ref = r_coordinates.xyzReader(self.dir + '/test_traj_pos_pbc.xyz')
        data_r, symbols_r, comments_r = _ref
        data, symbols, comments = copy.deepcopy(_ref)
        w_coordinates.xyzWriter(self.dir + '/out.xyz', data, symbols,
                                comments=comments)
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')

        self.assertTupleEqual(symbols_r, symbols2)
        self.assertListEqual(comments_r, comments2)
        self.assertTrue(np.array_equal(data_r, data2))

        # --- selection
        w_coordinates.xyzWriter(self.dir + '/out.xyz',
                                data,
                                symbols,
                                comments=comments,
                                selection=[1, 2])
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')
        self.assertTupleEqual(symbols_r[slice(1, 3)], symbols2)
        self.assertTrue(np.array_equal(data_r[:, 1:3], data2))

        os.remove(self.dir + "/out.xyz")

    def test_arcWriter(self):
        _ref = r_coordinates.arcReader(self.dir + '/water.arc')
        data_r, symbols_r, indices_r, types_r, connectivity_r, comments_r = \
            _ref
        data, symbols, indices, types, connectivity, comments = \
            copy.deepcopy(_ref)
        w_coordinates.arcWriter(self.dir + '/out.arc', data, symbols, types,
                                connectivity, comments)
        data2, symbols2, indices2, types2, connectivity2, comments2 = \
            r_coordinates.arcReader(self.dir + '/out.arc')

        self.assertTupleEqual(symbols_r, symbols2)
        self.assertTupleEqual(types_r, types2)
        self.assertTupleEqual(connectivity_r, connectivity2)
        self.assertTupleEqual(indices_r, indices2)
        self.assertListEqual(comments_r, comments2)
        self.assertTrue(np.array_equal(data_r, data2))

        # --- units
        w_coordinates.arcWriter(self.dir + '/out.arc', data, symbols, types,
                                connectivity, comments, units='default')
        data2, symbols2, indices2, types2, connectivity2, comments2 = \
            r_coordinates.arcReader(self.dir + '/out.arc')
        self.assertTrue(np.allclose(data_r, data2))
        w_coordinates.arcWriter(self.dir + '/out.arc', data, symbols, types,
                                connectivity, comments,
                                units=3*[('length', 'au')])
        data2, symbols2, indices2, types2, connectivity2, comments2 = \
            r_coordinates.arcReader(self.dir + '/out.arc')
        self.assertTrue(np.allclose(data_r, data2*constants.l_au2aa,
                                    atol=1.E-6))

        # --- selection
        w_coordinates.arcWriter(self.dir + '/out.arc', data, symbols, types,
                                connectivity, comments, selection=[2, 1])
        data2, symbols2, indices2, types2, connectivity2, comments2 = \
            r_coordinates.arcReader(self.dir + '/out.arc')

        self.assertTupleEqual(symbols_r[slice(2, 0, -1)], symbols2)
        self.assertTupleEqual(types_r[slice(2, 0, -1)], types2)
        self.assertTupleEqual(connectivity_r[slice(2, 0, -1)], connectivity2)
        self.assertTupleEqual(indices_r[slice(2, 0, -1)], indices2)
        self.assertTrue(np.allclose(data_r[:, [2, 1]], data2))

        os.remove(self.dir + "/out.arc")

    def test_pdbWriter(self):
        _ref = r_coordinates.pdbReader(self.dir + '/test_simple.pdb')
        data_r, names_r, symbols_r, res_r, cell_aa_deg_r, title_r = \
            _ref
        data, names, symbols, res, cell_aa_deg, title = \
            copy.deepcopy(_ref)

        # --- only single frames for now
        w_coordinates.pdbWriter(self.dir + '/out.pdb', data[0], names, symbols,
                                res, cell_aa_deg, title)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data2, names2, symbols2, res2, cell_aa_deg2, title2 =\
                r_coordinates.pdbReader(self.dir + '/out.pdb')

        self.assertTupleEqual(names_r, names2)
        self.assertTupleEqual(symbols_r, symbols2)
        self.assertTupleEqual(res_r, res2)
        self.assertTrue(np.array_equal(data_r, data2))
        self.assertListEqual(cell_aa_deg_r.tolist(), cell_aa_deg2.tolist())

        # --- selection
        w_coordinates.pdbWriter(self.dir + '/out.pdb', data[0], names, symbols,
                                res, cell_aa_deg, title, selection=[3, 5, 7])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data2, names2, symbols2, res2, cell_aa_deg2, title2 =\
                r_coordinates.pdbReader(self.dir + '/out.pdb')

        self.assertTupleEqual(names_r[slice(3, 9, 2)], names2)
        self.assertTupleEqual(symbols_r[slice(3, 9, 2)], symbols2)
        self.assertTupleEqual(res_r[slice(3, 9, 2)], res2)
        self.assertTrue(np.array_equal(data_r[:, slice(3, 9, 2)], data2))
        os.remove(self.dir + "/out.pdb")

        # check missing-CRYST1 warning (important feature in ChirPy)
        with self.assertWarns(ChirPyWarning):
            w_coordinates.pdbWriter(self.dir + '/out.pdb',
                                    data[0],
                                    names,
                                    symbols,
                                    res,
                                    np.array([0., 0., 0., 90., 90., 90.]),
                                    title)
        os.remove(self.dir + "/out.pdb")


class TestGrid(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cubeWriter(self):
        _ref = r_grid.cubeReader(self.dir + '/test-1.cube')
        data_r, origin_aa_r, cell_vec_aa_r, pos_aa_r, numbers_r, comments_r = \
            _ref
        data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
            copy.deepcopy(_ref)
        w_grid.cubeWriter(self.dir + '/out.cube',
                          comments[0],
                          numbers,
                          pos_aa[0],
                          cell_vec_aa,
                          data[0],
                          origin_aa=origin_aa)
        data2, origin_aa2, cell_vec_aa2, pos_aa2, numbers2, comments2 = \
            r_grid.cubeReader(self.dir + '/out.cube')

        self.assertTrue(np.array_equal(data_r, data2))
        self.assertTrue(np.array_equal(origin_aa_r, origin_aa2))
        self.assertTrue(np.array_equal(cell_vec_aa_r, cell_vec_aa2))
        self.assertTrue(np.array_equal(pos_aa_r, pos_aa2))
        self.assertTupleEqual(numbers_r, numbers2)
        self.assertListEqual(comments_r, comments2)
        os.remove(self.dir + "/out.cube")
