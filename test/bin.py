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

    def test_correlation_vibrational(self):
        os.system(
            'CORRELATION_CalculateVibrationalSpectra.py %s/MOL2 ' % self.dir
            + '--cell_aa_deg 12.072 12.342 11.576 90.00 90.00 90.00 '
            + '--ts 4 --return_tcf --va --vcd --cutoff 0 --filter_strength -1 '
            + '--xrange 2000 500 --save --noplot'
            )

        for _file in ['tcf_va_spectrum.dat', 'tcf_vcd_spectrum.dat',
                      'va_spectrum.dat', 'vcd_spectrum.dat']:
            self.assertTrue(
               filecmp.cmp(_file, self.dir + '/' + _file, shallow=False),
               f'Spectrum/TCF reproduced incorrectly: {self.dir + "/" +_file})'
               )
            os.remove(_file)
    # def test_correlation_power(self):
