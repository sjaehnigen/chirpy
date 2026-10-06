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
import filecmp

from chirpy.create import supercell

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_files'


class TestSupercell(unittest.TestCase):
    # --- ToDo: insufficiently tested

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

        try:
            sys._fill_box(verbose=False, sort_atoms=True)
            os.remove('topology.pdb')
            os.remove('packmol.inp')
            os.remove('packmol.log')

        except ImportError:
            os.remove('.member-000.pdb')
            os.remove('.member-001.pdb')
            os.remove('packmol.inp')
            os.remove('packmol.log')
            raise AssertionError("Could not find packmol installation. "
                                 "Some features of create module not "
                                 "available.")

    def test_molecular_crystal(self):
        c = supercell.MolecularCrystal(self.dir + '/782512.pdb')

        b = c.create(verbose=False, multiply=(1, 2, 2))
        # b.sort_atoms()
        b.wrap()
        b.write('CREATE.pdb')

        self.assertTrue(filecmp.cmp("CREATE.pdb",
                                    self.dir + "/782512_1x2x2.pdb",
                                    shallow=False),
                        f'{self.dir}/782512_1x2x2.pdb incorrectly reproduced'
                        ' by CREATE.pdb',
                        )

        os.remove('CREATE.pdb')

    def test_molecular_crystal_propagate_cell(self):
        # --- propagate() must scale the (possibly non-tetragonal) unit
        #     cell dimensions consistently with the atomic supercell
        #     expansion, keeping angles unchanged
        import numpy as np
        c = supercell.MolecularCrystal(self.dir + '/782512.pdb')
        cell0 = np.array(c.cell_aa_deg)

        b = c.create(verbose=False, multiply=(1, 2, 2))

        self.assertTrue(np.allclose(c.cell_aa_deg,
                                    cell0 * [1, 2, 2, 1, 1, 1]))
        # --- angles (incl. the non-90 deg beta angle) remain unchanged
        self.assertTrue(np.allclose(c.cell_aa_deg[3:], cell0[3:]))
        # --- the propagated frame's cell is kept in sync
        self.assertTrue(np.allclose(b.cell_aa_deg, c.cell_aa_deg))
