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

import unittest
import os
import numpy as np
import importlib
import pkgutil

# --- Done
from chirpy.read import modes as r_modes
from chirpy.read import coordinates as r_coordinates
from chirpy.read import grid as r_grid

from chirpy.write import modes as w_modes
# from chirpy.write import coordinates as w_coordinates
# from chirpy.write import grid as w_grid

from chirpy.interface import cpmd
# --- ToDo (change import path later after moving bin)
# from .topology import
# from .mathematics import
# from .statistics import
# from .physics import
# from .classes import
# classes: also insert true negatives (such as wrong or unknown format)
# from .interface import
# from .create import
# from .visualisation import
# from .mdanalysis import
# read iterators

# --- NOT tested
#
# snippets
# bin
# external
# visualisation
# hidden methods in general


# logger = debug.logger

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestImports(unittest.TestCase):

    @staticmethod
    def import_submodules(package, recursive=True):
        '''https://stackoverflow.com/a/25562415'''
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + '.' + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(TestImports.import_submodules(full_name))
        return results

    def test_import_00_read(self):
        self.import_submodules('chirpy.read')

    def test_import_00_write(self):
        self.import_submodules('chirpy.write')

    def test_import_00_topology(self):
        self.import_submodules('chirpy.topology')

    def test_import_00_mathematics(self):
        self.import_submodules('chirpy.mathematics')

    def test_import_00_physics(self):
        self.import_submodules('chirpy.physics')

    def test_import_01_interface(self):
        # numpy warning for scipy.interpolate import UnivariateSpline in vmd
        self.import_submodules('chirpy.interface')

    def test_import_01_mdanalysis(self):
        self.import_submodules('chirpy.mdanalysis')

    def test_import_02_classes(self):
        self.import_submodules('chirpy.classes')

    def test_import_03_create(self):
        self.import_submodules('chirpy.create')

    def test_import_10_visualise(self):
        self.import_submodules('chirpy.visualise')

    def test_import_10_external(self):
        self.import_submodules('chirpy.external')


class TestReaders(unittest.TestCase):

    def setUp(self):
        # Change paths after moving file
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

    def test_xyzReader(self):
        data, symbols, comments = \
                r_coordinates.xyzReader(self.dir + '/test_frame_pos_pbc.xyz')
        self.assertIsInstance(data, np.ndarray)
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_frame_pos_pbc').reshape(1, 208, 3)
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
                range=(1, 2, 4)
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
            tmp = tuple([(int(_a[0]), str(_a[1])) for _a in tmp])
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

        # check missing-CRYST1 warning (important feature in ChirPy)
        with self.assertWarns(RuntimeWarning):
            data, names, symbols, res, cell_aa_deg, title = \
                    r_coordinates.pdbReader(
                          self.dir + '/test_simple_nodims.pdb')

    def test_cubeReader(self):
        data, origin_au, cell_vec_au, pos_au, numbers, comments = \
                r_grid.cubeReader(self.dir + '/test-1.cube')
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(origin_au, np.ndarray)
        self.assertIsInstance(cell_vec_au, np.ndarray)
        self.assertIsInstance(pos_au, np.ndarray)
        self.assertIsInstance(comments, list)
        [self.assertIsInstance(_c, str) for _c in comments]
        self.assertTupleEqual(
                numbers,
                (6, 6, 6, 6, 6, 7, 1, 1, 1, 1, 1)
                )
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_volume_1').reshape(1, 6, 6, 6)
            ))

        data, origin_au, cell_vec_au, pos_au, numbers, comments = \
            r_grid.cubeReader(self.dir + '/test-2.cube')
        self.assertTupleEqual(data.shape, (1, 10, 10, 10))
        data, origin_au, cell_vec_au, pos_au, numbers, comments = \
            r_grid.cubeReader(self.dir + '/test-4.cube')
        self.assertTupleEqual(data.shape, (2, 6, 6, 6))

        # Some Negatives
        with self.assertRaises(ValueError):
            data, origin_au, cell_vec_au, pos_au, numbers, comments = \
                    r_grid.cubeReader(self.dir + '/test-3.cube')


class TestWriters(unittest.TestCase):

    def setUp(self):
        # Change paths after moving file
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

    # def test_xyzWriter(self):
    #     pass

    # def test_cpmdWriter(self):
    #     pass

    # def test_pdbWriter(self):
    #     pass

    # def test_cubeWriter(self):
    #     pass


class TestTopology(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestMathematics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestStatistics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestPhysics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestClasses(unittest.TestCase):
    # Later for trajectory class: load vels, find unknown elements

    def setUp(self):
        pass

    def tearDown(self):
        pass

# also test the Iterators (important if they return correct shapes, especially
# CPMD)

#    @unittest.expectedFailure
#    def test_fail(self):
#        self.assertEqual(1, 0, "broken")


class TestInterfaces(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cpmdReader(self):
        for _i, _n in zip(['GEOMETRY', 'MOMENTS', 'TRAJECTORY'],
                          [(1, 208, 6), (5, 288, 9), (6, 208, 6)]):

            data = cpmd.cpmdReader(self.dir + '/' + _i,
                                   filetype=_i,
                                   kinds=['X']*_n[1])['data']

            self.assertTrue(np.array_equal(
                data,
                np.genfromtxt(self.dir + '/data_' + _i).reshape(_n)
                ))

        # Some Negatives
        with self.assertRaises(ValueError):
            data = cpmd.cpmdReader(self.dir + '/MOMENTS_broken',
                                   filetype='MOMENTS',
                                   kinds=['X']*288)['data']
            data = cpmd.cpmdReader(self.dir + '/MOMENTS',
                                   filetype='MOMENTS',
                                   kinds=['X']*286)['data']
        # Test range
        data = cpmd.cpmdReader(self.dir + '/' + _i,
                               filetype='TRAJECTORY',
                               kinds=['X']*_n[1],
                               range=(2, 3, 8),
                               )['data']
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_TRAJECTORY').reshape(_n)[2:8:3]
            ))


class TestGenerators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


# Test the binaries (all arguments)

class TestVisualisation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestMdanalysis(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestTest:
    def setUp(self):
        # Change paths after moving file
        self.dir = _test_dir

    def tearDown(self):
        pass

    def test_test(self):
        pass


if __name__ == '__main__':
    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))
    unittest.main()
