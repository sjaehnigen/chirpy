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

# import logging
# import filecmp
# import types

# --- Done

# --- ToDo (change import path later after moving bin)
from readers import modes, trajectory, volume
# from .writers import
# from .topology import
# from .mathematics import
# from .statistics import
# from .physics import
# from .classes import
# classes: also insert true negatives (such as wrong or unknown format)
# from .interfaces import
# from .generators import
# from .visualisation import
# from .mdanalysis import

# --- NOT tested
#
# snippets
# bin
# external
# visualisation
# hidden methods in general


# logger = debug.logger

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestReaders(unittest.TestCase):

    def setUp(self):
        # Change paths after moving file
        self.dir = _test_dir + '/readers'

    def tearDown(self):
        pass

    def test_xvibsReader(self):
        n_atoms, numbers, pos_aa, n_modes, freqs, vec = modes.xvibsReader(self.dir + '/test.xvibs')
        self.assertEqual(n_atoms, 13)
        self.assertEqual(n_modes, 39)
        self.assertListEqual(numbers, [16, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertListEqual(freqs, [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            -1423.778846, 268.439735, 440.530004, 480.830349, 539.719158, 592.156510, 724.732367,
            774.372888, 787.075915, 939.734940, 951.363587, 976.477230, 977.793454, 988.142121,
            1069.087407, 1106.292445, 1151.102945, 1224.764529, 1246.010705, 1336.580029,
            1385.798005, 1404.196670, 1472.650641, 1482.110876, 1492.012368, 1530.493063,
            3017.469174, 3018.251568, 3053.243822, 3056.060332, 3075.051383, 3080.396635,
            3096.470496])
        [self.assertIsInstance(_m, np.ndarray) for _m in (pos_aa, vec)]
        self.assertTrue(np.array_equal(pos_aa, np.array([[-2.742036245848,  2.530103531747,   -0.248174312943],
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
                                                      [-4.947076501132, -0.047031588011,    0.365545752725],])))
        self.assertTrue(np.array_equal(vec, np.genfromtxt(self.dir + '/vec').reshape(39, 13, 3)))

        #ToDo: I have to make sure that the test file fulfils my xvibs standard

    def test_xyzReader(self):
        data, symbols, comments = trajectory.xyzReader(self.dir + '/test_frame.xyz')
        data, symbols, comments = trajectory.xyzReader(self.dir + '/test_traj.xyz')
        # verify types: np.array, np.array, list and shapes


        # self.assertTrue((at_raman[1] == ref1).all())
        # self.assertTrue((at_vcd[4] == ref2).all())
        # self.assertTrue(np.array([(a-b)/a < mp for a, b in zip(ref3.flat, at_test.flat)]).all())
        # self.assertEqual(widget.size(), (50, 50))
        # assertIsInstance(a, b)
        # assertIsNone(x)
        # assertIsNot(a, b)
        # assertIn(a, b)
        # assertLessEqual(a, b)
        # assertMultiLineEqual()

        # # check that s.split fails when the separator is not a string
        # with self.assertRaises(TypeError):
        #     s.split(2)
    # def test_cpmdReader(self):
    #     data = trajectory.cpmdReader(self.dir + '/GEOMETRY')
    #     data = trajectory.cpmdReader(self.dir + '/MOMENTS')
    #     data = trajectory.cpmdReader(self.dir + '/TRAJECTORY')
    #     # verify types: np.array and shape

    # def test_pdbReader(self):
    #     data, names, symbols, res, cell_aa_deg, title = trajectory.pdbReader(self.dir + '/test.pdb')
    #     # verify types: np.array, np.array?, np.array, np.array, np.array, string? and shapes

    # def test_cubeReader(self):
    #     data = volume.cubeReader(self.dir + '/test.cube')


class TestWriters(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


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

    def setUp(self):
        pass

    def tearDown(self):
        pass

#    @unittest.expectedFailure
#    def test_fail(self):
#        self.assertEqual(1, 0, "broken")



class TestInterfaces(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestGenerators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


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


if __name__ == '__main__':
    unittest.main()
