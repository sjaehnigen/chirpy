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
import numpy as np
import warnings

from chirpy.read import modes as r_modes
from chirpy.read import coordinates as r_coordinates
from chirpy.read import grid as r_grid
from chirpy.read import generators

from chirpy import constants
from chirpy.config import ChirPyWarning

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestModes(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_xvibsReader(self):
        n_atoms, numbers, pos_aa, n_modes, freqs, modes = \
                r_modes.xvibsReader(self.dir + '/test.xvibs')
        self.assertEqual(n_atoms, 13)
        self.assertEqual(n_modes, 39)
        self.assertListEqual(numbers, [16, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertListEqual(
              freqs,
              [0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000,
               -1423.778846, 268.439735, 440.530004, 480.830349, 539.719158,
               592.156510, 724.732367, 774.372888, 787.075915, 939.734940,
               951.363587, 976.477230, 977.793454, 988.142121, 1069.087407,
               1106.292445, 1151.102945, 1224.764529, 1246.010705, 1336.580029,
               1385.798005, 1404.196670, 1472.650641, 1482.110876, 1492.012368,
               1530.493063, 3017.469174, 3018.251568, 3053.243822, 3056.060332,
               3075.051383, 3080.396635, 3096.470496]
              )
        [self.assertIsInstance(_m, np.ndarray) for _m in (pos_aa, modes)]
        self.assertTrue(np.array_equal(
              pos_aa,
              np.array([[-2.742036245848,  2.530103531747,   -0.248174312943],
                        [-3.910785516958,  1.814433400469,    0.126044209092],
                        [-4.035315310420,  0.424972102488,    0.001169531098],
                        [-2.903463572288, -0.377120126662,   -0.191881605576],
                        [-1.641786496853,  0.136803402044,   -0.594842102838],
                        [-1.773415317816,  1.470517109583,   -0.038523347977],
                        [-2.945726976286, -1.382631165206,    0.232506321704],
                        [-1.546865436278,  0.605531600423,   -1.575057529439],
                        [-0.764991609146, -0.467964892707,   -0.346968966401],
                        [-2.589894422022,  3.502773637521,    0.227939482068],
                        [-2.436470963052,  2.538885880037,   -1.295163431997],
                        [-4.642467810246,  2.309392171426,    0.768382988060],
                        [-4.947076501132, -0.047031588011,    0.365545752725]])
              ))
        self.assertTrue(np.array_equal(
            modes,
            np.genfromtxt(self.dir + '/vec').reshape(39, 13, 3)
            ))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            n_atoms, numbers, pos_aa, n_modes, freqs, modes_mw = \
                r_modes.xvibsReader(self.dir + '/test.xvibs', mw=True)
            self.assertTrue(np.allclose(
                modes_mw * np.sqrt(constants.numbers_to_masses(
                                   numbers))[None, :, None],
                modes,
                ))


class TestCoordinates(unittest.TestCase):

    def setUp(self):
        # Change paths after moving file
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_xyzReader(self):
        # --- general iterator test (also valid for cpmd, cube, etc.)
        data, symbols, comments = \
                r_coordinates.xyzReader(self.dir + '/test_traj_w_doublets.xyz',
                                        skip=[9, 10, 11, 12], range=(0, 3, 12))
        self.assertTupleEqual(data.shape, (4, 1, 3))
        self.assertTrue(all(['doublet' not in _c for _c in comments]))

        # --- xyz
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_frame_pos_pbc.xyz')
        self.assertIsInstance(data, np.ndarray)
        DATA_FRAME_POS_PBC = np.loadtxt(
                self.dir + '/data_frame_pos_pbc'
                ).reshape(1, 208, 3)
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))
        self.assertIsInstance(comments, list)
        self.assertTupleEqual(
                symbols,
                ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
                 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', )
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        data, symbols, comments = r_coordinates.xyzReader(
                self.dir + '/test_frame_posvel_pbc.xyz')
        self.assertIsInstance(data, np.ndarray)
        self.assertTrue(np.array_equal(
          data,
          np.genfromtxt(self.dir + '/data_frame_posvel_pbc').reshape(1, 208, 6)
          ))
        self.assertIsInstance(comments, list)
        self.assertTupleEqual(
                symbols,
                ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
                 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', )
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        data, symbols, comments = r_coordinates.xyzReader(
                self.dir + '/test_traj_pos_pbc.xyz')
        self.assertIsInstance(data, np.ndarray)
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_traj_pos_pbc').reshape(3, 393, 3)
            ))
        self.assertIsInstance(comments, list)
        self.assertTupleEqual(
                symbols,
                ('C', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'O', 'H', 'C',
                 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'O', 'H', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H',
                 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', )
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        # --- test unit conversion (also valid for cpmd, arc)
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_frame_pos_pbc.xyz',
                                    units=3*[('length', 'aa')])
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_frame_pos_pbc.xyz',
                                    units='default')
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_frame_pos_pbc.xyz',
                                    units=3*[('length', 'au')])
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC *
                        constants.l_au2aa))

        # Test symbol support (no recognition here)
        data, symbols, comments = r_coordinates.xyzReader(
                self.dir + '/test_symbols.xyz')
        self.assertTupleEqual(
                symbols,
                ('H', 'C', 'C', 'O', 'C', 'C', 'He', 'C', 'C', 'D', 'C', 'Xe',
                 'C', 'S', 'C', 'N', 'C', 'P', 'Al', 'C', 'I', 'Cl', 'F', 'C',
                 'C', 'Mg', 'C', 'Na')
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        # Some Negatives
        with self.assertRaises(ValueError):
            data, symbols, comments = r_coordinates.xyzReader(
                    self.dir + '/test_wrong_number.xyz')
            data, symbols, comments = r_coordinates.xyzReader(
                    self.dir + '/test_broken_frame.xyz')
            data, symbols, comments = r_coordinates.xyzReader(
                    self.dir + '/test_broken_frame_2.xyz')

        # Test range
        data, symbols, comments = r_coordinates.xyzReader(
                self.dir + '/test_traj_pos_pbc.xyz',
                range=(1, 1, 3)
                )
        self.assertTrue(np.array_equal(
         data,
         np.genfromtxt(self.dir + '/data_traj_pos_pbc').reshape(3, 393, 3)[1:3]
         ))

        data, symbols, comments = r_coordinates.xyzReader(
                self.dir + '/test_traj_pos_pbc.xyz',
                range=(1, 2, 3)
                )
        self.assertTrue(np.array_equal(
         data,
         np.genfromtxt(self.dir +
                       '/data_traj_pos_pbc').reshape(3, 393, 3)[1:4:2]
         ))

    def test_pdbReader(self):
        # not much testing of protein features as this is an external reader
        data, names, symbols, res, cell_aa_deg, title = \
            r_coordinates.pdbReader(self.dir + '/test_protein.pdb')
        data, names, symbols, res, cell_aa_deg, title = \
            r_coordinates.pdbReader(self.dir + '/test_raw.pdb')
        data, names, symbols, res, cell_aa_deg, title =\
            r_coordinates.pdbReader(self.dir + '/test_simple.pdb')

        self.assertTrue(np.allclose(cell_aa_deg, np.array([
            12.072, 12.342, 11.576, 90.00, 90.00, 90.00])))

        with open(self.dir + '/res', 'r') as _f:
            tmp = [tuple(_l.strip().split('-')) for _l in _f.readlines()]
            tmp = tuple([[int(_a[0]), str(_a[1])] for _a in tmp])
        self.assertTupleEqual(res, tmp)

        self.assertTupleEqual(
                symbols,
                ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
                 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', )
                )

    def test_cifReader(self):
        # ToDo: insufficient testing: needs test on symmetry
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            data, names, symbols, cell_aa_deg, title = \
                r_coordinates.cifReader(self.dir + '/indanol.cif')
        self.assertTupleEqual(data.shape, (1, 40, 3))
        self.assertIsInstance(title, list)
        self.assertTupleEqual(
                symbols,
                ('C', 'H', 'C', 'H', 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'H',
                 'C', 'H', 'C', 'H', 'C', 'H', 'O', 'H', 'C', 'H', 'C', 'H',
                 'H', 'C', 'H', 'H', 'C', 'C', 'C', 'H', 'C', 'H', 'C', 'H',
                 'C', 'H', 'O', 'H')
                )
        [self.assertIsInstance(_c, str) for _c in title]

        self.assertListEqual(
            np.around(data, decimals=12).flatten().tolist(),
            np.genfromtxt(self.dir + '/indanol_ref').tolist()
          )

    def test_arcReader(self):
        # --- xyz
        # --- Check Fortran number conversion
        data, symbols, numbers, types, connectivity, comments = \
            r_coordinates.arcReader(self.dir + '/water.vel')
        self.assertTrue(np.allclose(
            data,
            np.genfromtxt(self.dir + '/tinker_vel').reshape(10, 3, 3),
            atol=1.E-12
            ))
        data, symbols, numbers, types, connectivity, comments = \
            r_coordinates.arcReader(self.dir + '/water.arc')
        self.assertTrue(np.allclose(
            data,
            np.genfromtxt(self.dir + '/tinker_pos').reshape(10, 3, 3),
            atol=1.E-12
            ))
        self.assertIsInstance(comments, list)
        self.assertTupleEqual(
                symbols,
                ('O', 'H', 'H')
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        data, symbols, numbers, types, connectivity, comments = \
            r_coordinates.arcReader(self.dir + '/TAI.xyz')

        self.assertTupleEqual(
                connectivity,
                (
                    [2, 6, 11],
                    [1, 3, 13],
                    [2, 4, 7],
                    [3, 5, 8],
                    [4, 6, 9],
                    [1, 5, 10],
                    [3],
                    [4],
                    [5],
                    [6],
                    [1, 12, 16, 19],
                    [11],
                    [2, 14, 15, 16],
                    [13],
                    [13],
                    [11, 13, 17, 21],
                    [16],
                    [19],
                    [11, 18, 20],
                    [19],
                    [16, 22],
                    [21]
                    )
                )
        self.assertTupleEqual(
                types,
                (304, 305, 303, 302, 301, 300, 310, 309, 308, 307, 59, 60, 57,
                 58, 58, 55, 56, 66, 65, 66, 230, 231)
                )

        # test CELL
        # data, symbols, numbers, types, connectivity, comments, cell_aa_deg
        _return = r_coordinates.arcReader(self.dir + '/s_0881.arc')
        self.assertEqual(len(_return), 7)
        self.assertListEqual(
                _return[0].ravel().tolist(),
                np.loadtxt(self.dir + '/data_s_0881_arc').tolist()
                )
        self.assertListEqual(_return[-1].tolist(),
                             [12.073, 12.342, 11.577, 90., 90., 90.])
        # Some Negatives
        with self.assertRaises(ValueError):
            _return = r_coordinates.arcReader(self.dir + '/s_0881_broken.arc')

    def test_freeIterator(self):
        reference = iter([
            [0.255269, 1.97165, -14.108334],
            [1., 0.255269, 1.97165, -14.108334],
            [1.97165, -14.108334],
            [3., 1., 0.255269, 1.97165, -14.108334],
        ])
        for columns in ['imddd', 'idddd', 'ismdd', 'ddddd']:
            _return = r_coordinates.freeIterator(self.dir+'/s_0881.dip',
                                                 columns=columns)
            data = next(_return)
            self.assertEqual(len(data), len(set(columns)))
            self.assertListEqual(data[-1][0].tolist(), next(reference))
        # Some Negatives
        with self.assertRaises(ValueError):
            _return = r_coordinates.freeIterator(self.dir+'/s_0881.dip',
                                                 columns='iiiii')

    def test_container(self):
        _return = generators._container(
                (r_coordinates.freeIterator, r_coordinates.freeIterator),
                (self.dir + '/s_0881.dip', self.dir + '/s_0881.dip'),
                (),
                ({'columns': 'imddd'}, {'columns': 'imddd'}))
        data = next(_return)
        self.assertTupleEqual(data[1], 2*(tuple(np.arange(1, 17).astype(str)),)
                              )
        self.assertTupleEqual(np.array(data[2]).shape, (2, 16, 3))

    def test_xyzContainer(self):
        _iter = r_coordinates.xyzContainer(
                    self.dir + '/test_frame_pos_pbc.xyz',
                    self.dir + '/test_frame_pos_pbc.xyz',
                    self.dir + '/test_frame_pos_pbc.xyz',
                )
        data, symbols, comments = next(_iter)
        self.assertIsInstance(data, np.ndarray)
        DATA_FRAME_POS_PBC = np.tile(np.loadtxt(
                self.dir + '/data_frame_pos_pbc'
                ).reshape(208, 3), (1, 3))
        self.assertListEqual(data[0].tolist(), DATA_FRAME_POS_PBC[0].tolist())
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))
        self.assertTupleEqual(
                symbols,
                ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                 'H', 'H', 'H', 'H', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
                 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', )
                )
        [self.assertIsInstance(_c, str) for _c in comments]

        # --- test unit conversion (also valid for cpmd, arc)
        # --- some defaults (no change to numbers)
        data, symbols, comments = next(r_coordinates.xyzContainer(
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                units=9*[('length', 'aa')]
                ))
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))
        data, symbols, comments = next(r_coordinates.xyzContainer(
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                units=6*[('length', 'aa')] + 3*[('velocity', 'au')]
                ))
        self.assertTrue(np.array_equal(data, DATA_FRAME_POS_PBC))

        # --- non-defaults
        data, symbols, comments = next(r_coordinates.xyzContainer(
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                self.dir + '/test_frame_pos_pbc.xyz',
                units=6*[('length', 'aa')] + 3*[('length', 'au')]
                ))
        data[:, 6:] *= constants.l_aa2au
        self.assertListEqual(data[0].tolist(), DATA_FRAME_POS_PBC[0].tolist())
        self.assertTrue(np.allclose(data, DATA_FRAME_POS_PBC))

        # --- some Negatives
        with self.assertRaises(ValueError):
            data, symbols, comments = next(r_coordinates.xyzContainer(
                    self.dir + '/test_frame_pos_pbc.xyz',
                    self.dir + '/test_frame_pos_pbc.xyz',
                    self.dir + '/test_frame_pos_pbc.xyz',
                    units=6*[('length', 'aa')]
                    ))

    def test_arcContainer(self):
        TINKER_VEL = np.loadtxt(self.dir + '/tinker_vel').reshape(10, 3, 3)[0]
        TINKER_POS = np.loadtxt(self.dir + '/tinker_pos').reshape(10, 3, 3)[0]
        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.vel',
                self.dir + '/water.vel',
                ))
        self.assertTrue(np.allclose(
            data,
            np.tile(TINKER_VEL, (1, 2)),
            atol=1.E-12
            ))
        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.arc',
                self.dir + '/water.arc',
                ))
        self.assertTrue(np.array_equal(data, np.tile(TINKER_POS, (1, 2))))
        self.assertTupleEqual(
                symbols,
                ('O', 'H', 'H')
                )

        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/TAI.xyz',
                self.dir + '/TAI.xyz',
                ))

        self.assertTupleEqual(
                connectivity,
                (
                    [2, 6, 11],
                    [1, 3, 13],
                    [2, 4, 7],
                    [3, 5, 8],
                    [4, 6, 9],
                    [1, 5, 10],
                    [3],
                    [4],
                    [5],
                    [6],
                    [1, 12, 16, 19],
                    [11],
                    [2, 14, 15, 16],
                    [13],
                    [13],
                    [11, 13, 17, 21],
                    [16],
                    [19],
                    [11, 18, 20],
                    [19],
                    [16, 22],
                    [21]
                    )
                )
        self.assertTupleEqual(
                types,
                (304, 305, 303, 302, 301, 300, 310, 309, 308, 307, 59, 60, 57,
                 58, 58, 55, 56, 66, 65, 66, 230, 231)
                )

        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.arc',
                self.dir + '/water.vel',
                ))
        self.assertTrue(np.allclose(data,
                                    np.hstack((TINKER_POS, TINKER_VEL)),
                                    atol=1.E-12))

        # ---- units
        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.arc',
                self.dir + '/water.vel',
                units='default',
                ))
        self.assertTrue(np.allclose(data,
                                    np.hstack((TINKER_POS, TINKER_VEL)),
                                    atol=1.E-12))

        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.arc',
                self.dir + '/water.vel',
                units=3*[('length', 'aa')] + 3*[('velocity', 'aa_ps')],
                ))
        self.assertTrue(np.allclose(data,
                                    np.hstack((TINKER_POS, TINKER_VEL)),
                                    atol=1.E-12))

        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates.arcContainer(
                self.dir + '/water.arc',
                self.dir + '/water.vel',
                units=3*[('length', 'aa')] + 3*[('velocity', 'aa_fs')],
                ))
        self.assertTrue(np.allclose(
                    data,
                    np.hstack((TINKER_POS, TINKER_VEL*1000.)),
                    atol=1.E-12)
                    )

    def test_mixedContainer(self):
        TINKER_VEL = np.loadtxt(self.dir + '/tinker_vel').reshape(10, 3, 3)[0]
        TINKER_POS = np.loadtxt(self.dir + '/tinker_pos').reshape(10, 3, 3)[0]
        data, symbols, numbers, types, connectivity, comments = \
            next(r_coordinates._coordContainer(
                self.dir + '/water.xyz',
                self.dir + '/water.vel',
                iterator=[r_coordinates.xyzIterator,
                          r_coordinates.arcIterator],
                ))
        self.assertTrue(np.allclose(data,
                                    np.hstack((TINKER_POS, TINKER_VEL)),
                                    atol=1.E-12))

    def test_bz2(self):
        # --- general iterator test (also valid for cpmd, cube, etc.)
        rdata, rsymbols, rcomments = \
                r_coordinates.xyzReader(self.dir + '/test_long_mol.xyz')
        data, symbols, comments = \
            r_coordinates.xyzReader(self.dir + '/test_long_mol.xyz.bz2')
        [self.assertIsInstance(_c, str) for _c in comments]
        self.assertTrue(np.array_equal(data, rdata))
        self.assertTupleEqual(symbols, rsymbols)
        self.assertListEqual(comments, rcomments)


class TestGrid(unittest.TestCase):

    def setUp(self):
        # Change paths after moving file
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cubeReader(self):
        data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
                r_grid.cubeReader(self.dir + '/test-1.cube')
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(origin_aa, np.ndarray)
        self.assertIsInstance(cell_vec_aa, np.ndarray)
        self.assertIsInstance(pos_aa, np.ndarray)
        self.assertIsInstance(comments, list)
        [self.assertIsInstance(_c, tuple) for _c in comments]
        self.assertTupleEqual(
                numbers,
                (6, 6, 6, 6, 6, 7, 1, 1, 1, 1, 1)
                )
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_volume_1').reshape(1, 6, 6, 6)
            ))
        self.assertListEqual((origin_aa*constants.l_aa2au).tolist(),
                             [-10.507273, -8.971296, -12.268080]
                             )

        data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
            r_grid.cubeReader(self.dir + '/test-2.cube')
        self.assertTupleEqual(data.shape, (1, 10, 10, 10))
        data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
            r_grid.cubeReader(self.dir + '/test-4.cube')
        self.assertTupleEqual(data.shape, (2, 6, 6, 6))

        # Some Negatives
        with self.assertRaises(ValueError):
            data, origin_aa, cell_vec_aa, pos_aa, numbers, comments = \
                    r_grid.cubeReader(self.dir + '/test-3.cube')
