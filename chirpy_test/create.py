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
import filecmp

from chirpy.create import supercell

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

        sys._fill_box(verbose=False, sort_atoms=True)

        os.remove('topology.pdb')
        os.remove('packmol.inp')
        os.remove('packmol.log')

    def test_molecular_crystal(self):
        c = supercell.MolecularCrystal(self.dir + '/782512.pdb')

        b = c.create(verbose=False, multiply=(1, 2, 2))
        b.sort_atoms()
        b.wrap_atoms()
        b.write('out.pdb')

        self.assertTrue(filecmp.cmp("out.pdb",
                                    self.dir + "/782512_1x2x2.pdb",
                                    shallow=False),
                        'Creator does not reproduce reference file '
                        '782512_1x2x2.pdb (see out.pdb)!',
                        )

        os.remove('out.pdb')