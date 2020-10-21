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
import filecmp

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestBinaries(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/interface_bin'

    def tearDown(self):
        pass

    def test_cpmd_assemble_moments(self):
        os.system('CPMD_AssembleMoments.py %s/TOPOLOGY.pdb %s/TRAJECTORY '
                  '%s/MOMENTS -f TEST' % (3*(self.dir,)))
        self.assertTrue(filecmp.cmp('TEST',
                                    self.dir + '/MOL',
                                    shallow=False),
                        'Molecular moments reproduced incorrectly (see TEST)'
                        )
        os.remove('TEST')

    def test_trajectory_convert(self):
        os.system('TRAJECTORY_Convert.py %s/water.arc --fn_vel %s/water.vel '
                  % (2 * (self.dir + '/../read_write',)) +
                  '-f out.xyz')
        self.assertTrue(filecmp.cmp('out.xyz',
                                    self.dir + '/trajectory.xyz',
                                    shallow=False),
                        'Trajectory reproduced incorrectly (see out.xyz)'
                        )
        os.remove('out.xyz')
