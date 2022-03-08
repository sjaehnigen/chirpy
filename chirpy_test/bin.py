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
#  Copyright (c) 2010-2021, The ChirPy Developers.
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
# -------------------------------------------------------------------

import unittest
import os
import filecmp
import numpy as np
import chirpy as cp


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
                        f'Molecular moments {self.dir}/MOL reproduced '
                        'incorrectly in TEST'
                        )
        os.remove('TEST')
        os.system('MOMENTS_AssembleMolecules.py %s/TOPOLOGY_DIMER.pdb '
                  '%s/TRAJECTORY_DIMER '
                  '%s/MOMENTS_DIMER -f TEST_DIMER' % (3*(self.dir,)))
        self.assertTrue(filecmp.cmp('TEST_DIMER',
                                    self.dir + '/DIMER',
                                    shallow=False),
                        f'Molecular moments {self.dir}/DIMER reproduced '
                        'incorrectly in TEST_DIMER'
                        )
        os.remove('TEST_DIMER')

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
        def cp_comp(fn, ref):
            return np.allclose(
                    cp.read.coordinates.xyzReader(fn)[0],
                    cp.read.coordinates.xyzReader(ref)[0],
                    atol=1.E-12)

        os.system('TRAJECTORY_Convert.py %s/water.arc --fn_vel %s/water.vel '
                  % (2 * (self.dir + '/../read_write',)) +
                  '-o out.xyz')
        self.assertTrue(cp_comp('out.xyz', self.dir + '/trajectory.xyz'),
                        f'Trajectory {self.dir}/trajectory.xyz reproduced '
                        f'incorrectly in out.xyz'
                        )
        os.remove('out.xyz')

        os.system('TRAJECTORY_Convert.py ' +
                  '%s/water_rot.arc --fn_vel %s/water_rot.vel ' %
                  (2*(self.dir + '/../read_write',)) +
                  '-o out.xyz --align_coords True')
        self.assertTrue(cp_comp('out.xyz', self.dir+'/trajectory_aligned.xyz'),
                        f'Trajectory {self.dir}/trajectory_aligned.xyz '
                        f'reproduced incorrectly in out.xyz'
                        )
        os.remove('out.xyz')

        os.system('TRAJECTORY_Convert.py %s/water.arc --fn_vel %s/water.vel '
                  % (2 * (self.dir + '/../read_write',)) +
                  '-o out.xyz --mask_frames 1 3')

        self.assertTrue(cp_comp('out.xyz', self.dir+'/trajectory_skip.xyz'),
                        f'Trajectory {self.dir}/trajectory_skip.xyz reproduced'
                        ' incorrectly in out.xyz'
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
               np.allclose(
                           np.loadtxt(_file),
                           np.loadtxt(self.dir + '/' + _file),
                           atol=0.0
                           ),
               # self.assertTrue(
               #    filecmp.cmp(_file, self.dir + '/' + _file, shallow=False),
               f'Spectrum/TCF reproduced incorrectly: {self.dir + "/" +_file}'
               f' in {_file}.'
               )
            os.remove(_file)
    # def test_correlation_power(self):
