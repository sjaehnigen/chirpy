#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------
#!/usr/bin/python
import os
import unittest
import logging
import filecmp
import numpy as np
import types
import filecmp

from lib import debug
from lib import constants
# from fileio import basicio
from fileio import cpmd


logger = debug.logger


class TestInputRoutines(unittest.TestCase):


    def setUp(self):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.fn_out = '%s/data/test.out'%self.dir_path
        self.fn_at_raman = '%s/data/at_RAM-APT'%self.dir_path
        self.fn_at_vcd = '%s/data/at_VCD-AT'%self.dir_path
        self.fn_at_test = '%s/data/at_test'%self.dir_path
        # self.documentation = [xyz.__dict__.get(a).__doc__ for a in dir(xyz)
        #                        if isinstance(xyz.__dict__.get(a), types.FunctionType)]


    def tearDown(self):
        if os.path.isfile(self.fn_out):
            os.remove(self.fn_out)


    # def test_Documentation(self):
    #     for d in self.documentation:
    #         self.assertTrue('Input' in d and 'Output' in d)


    def test_ReadATFile(self):
        ref1 = np.array([ 4.06853325453052639E-002, -3.12775953326507525E-002,  5.55562459513018192E-002,
                         -3.32416198986256045E-002,  2.01444925085307869E-002,  7.85000122066202288E-002,
                          5.75735956036268759E-002,  8.03152416226070864E-002, -6.44808209553398015E-002])
        ref2 = np.array([ 1.68860245877550837E-002, -2.96089324924537234E+000,  4.25953452509255737E+000,
                          3.12438368496530039E+000, -3.63585908457616722E-002, -1.65723610514223640E+000,
                         -4.48081200006128633E+000,  1.28926097105133808E+000,  3.15415600747256941E-008])
        mp = 1E-14
        ref3 = np.zeros((3, 9))
        for i in range(9):
            for j in range(3):
                k = i + 1
                ref3[j][i] = ((-1)**k * k + j * mp) * 1E100**j
        at_raman = cpmd.ReadATFile(self.fn_at_raman)
        at_vcd = cpmd.ReadATFile(self.fn_at_vcd)
        at_test = cpmd.ReadATFile(self.fn_at_test)
        self.assertTrue((at_raman[1] == ref1).all())
        self.assertTrue((at_vcd[4] == ref2).all())
        self.assertTrue(np.array([(a-b)/a < mp for a, b in zip(ref3.flat, at_test.flat)]).all())


if __name__ == '__main__':
  unittest.main()
#EOF