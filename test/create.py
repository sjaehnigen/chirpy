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

from ..create import supercell

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestSupercell(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/create'

    def tearDown(self):
        pass

    def test_solution(self):
        sys = supercell.Solution(
                        solvent=self.dir + '/ACN-d3.xyz',
                        solutes=[self.dir + '/S-MPAA.xyz'],
                        c_mol_L=[0.3],
                        rho_g_cm3=0.844,
                      )

        sys._fill_box(verbose=False, sort=True)

        os.remove('topology.pdb')
        os.remove('packmol.inp')
        os.remove('packmol.log')

    def test_molecular_crystal(self):
        c = supercell.MolecularCrystal(self.dir + '/782512.pdb')

        b = c.create(multiply=(1, 2, 2))
        b.sort_atoms()
        b.wrap_atoms()
        b.write('out.pdb')

        self.assertTrue(filecmp.cmp("out.pdb",
                                    self.dir + "/782512_1x2x2.pdb",
                                    shallow=False),
                        'Creator does not reproduce reference file '
                        '(see out.pdb)!',
                        )

        os.remove('out.pdb')
