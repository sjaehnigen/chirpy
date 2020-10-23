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

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestBinaries(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/interface_bin'

    def tearDown(self):
        pass

    def test_moments_assemble_molecules(self):
        os.system('MOMENTS_AssembleMolecules.py %s/TOPOLOGY.pdb %s/TRAJECTORY '
                  '%s/MOMENTS -f TEST' % (3*(self.dir,)))
        self.assertTrue(filecmp.cmp('TEST',
                                    self.dir + '/MOL',
                                    shallow=False),
                        'Molecular moments reproduced incorrectly (see TEST)'
                        )
        os.remove('TEST')

    def test_system_create_topology(self):
        for _f in ['topo-1.restart', 'topo.xyz']:
            os.system(f'SYSTEM_CreateTopologyFile.py {self.dir}/{_f}' +
                      ' --cell_aa_deg 25.520 25.520 25.520 90.00 90.00 90.00'
                      )
            self.assertTrue(filecmp.cmp('out.pdb',
                                        f'{self.dir}/topo.pdb',
                                        shallow=False),
                            f'Topology {self.dir}/topo.pdb reproduced '
                            f'incorrectly from {self.dir}/{_f}'
                            )
            filecmp.clear_cache()
            os.remove('out.pdb')

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
