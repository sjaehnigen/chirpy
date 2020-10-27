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
import numpy as np
import warnings

from chirpy.read import modes as r_modes
from chirpy.read import coordinates as r_coordinates
from chirpy.read import grid as r_grid

from chirpy.write import modes as w_modes
from chirpy.write import coordinates as w_coordinates
from chirpy.write import grid as w_grid


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
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_frame_posvel_pbc.xyz')
        w_coordinates.xyzWriter(self.dir + '/out.xyz', data, symbols, comments)
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')

        self.assertTupleEqual(symbols, symbols2)
        self.assertListEqual(comments, comments2)
        self.assertTrue(np.array_equal(data, data2))

        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_traj_pos_pbc.xyz')
        w_coordinates.xyzWriter(self.dir + '/out.xyz', data, symbols, comments)
        data2, symbols2, comments2 = \
            r_coordinates.xyzReader(self.dir + '/out.xyz')

        self.assertTupleEqual(symbols, symbols2)
        self.assertListEqual(comments, comments2)
        self.assertTrue(np.array_equal(data, data2))
        os.remove(self.dir + "/out.xyz")

    def test_arcWriter(self):
        data, symbols, indices, types, connectivity, comments = \
            r_coordinates.arcReader(self.dir + '/water.arc')
        w_coordinates.arcWriter(self.dir + '/out.arc', data, symbols, types,
                                connectivity, comments)
        data2, symbols2, indices2, types2, connectivity2, comments2 = \
            r_coordinates.arcReader(self.dir + '/out.arc')

        self.assertTupleEqual(symbols, symbols2)
        self.assertTupleEqual(types, types2)
        self.assertTupleEqual(connectivity, connectivity2)
        self.assertTupleEqual(indices, indices2)
        self.assertListEqual(comments, comments2)
        self.assertTrue(np.array_equal(data, data2))

        os.remove(self.dir + "/out.arc")

    def test_pdbWriter(self):
        data, names, symbols, res, cell_aa_deg, title =\
            r_coordinates.pdbReader(self.dir + '/test_simple.pdb')

        # --- only single frames for now
        w_coordinates.pdbWriter(self.dir + '/out.pdb', data[0], names, symbols,
                                res, cell_aa_deg, title)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data2, names2, symbols2, res2, cell_aa_deg2, title2 =\
                r_coordinates.pdbReader(self.dir + '/out.pdb')

        self.assertTupleEqual(names, names2)
        self.assertTupleEqual(symbols, symbols2)
        self.assertTupleEqual(res, res2)
        self.assertTrue(np.array_equal(data, data2))
        self.assertListEqual(cell_aa_deg.tolist(), cell_aa_deg2.tolist())
        os.remove(self.dir + "/out.pdb")


class TestGrid(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cubeWriter(self):
        data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
                r_grid.cubeReader(self.dir + '/test-1.cube')
        w_grid.cubeWriter(self.dir + '/out.cube',
                          comments[0],
                          numbers,
                          pos_aa[0],
                          cell_vec_aa,
                          data[0],
                          origin_aa=origin_aa)
        data2, origin_aa2, cell_vec_aa2, pos_aa2, numbers2, comments2 = \
            r_grid.cubeReader(self.dir + '/out.cube')

        self.assertTrue(np.array_equal(data, data2))
        self.assertTrue(np.array_equal(origin_aa, origin_aa2))
        self.assertTrue(np.array_equal(cell_vec_aa, cell_vec_aa2))
        self.assertTrue(np.array_equal(pos_aa, pos_aa2))
        self.assertTupleEqual(numbers, numbers2)
        self.assertListEqual(comments, comments2)
        os.remove(self.dir + "/out.cube")
