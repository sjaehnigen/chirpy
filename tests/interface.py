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
import numpy as np
import warnings
import filecmp
import copy

from chirpy.interface import cpmd, tinker
from chirpy import constants
from chirpy.config import ChirPyWarning

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestCPMD(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cpmdReader(self):
        for _i, _n in zip(['GEOMETRY', 'MOMENTS', 'TRAJECTORY'],
                          [(1, 208, 6), (5, 288, 9), (6, 208, 6)]):

            data = cpmd.cpmdReader(self.dir + '/' + _i,
                                   filetype=_i,
                                   symbols=['X']*_n[1])['data']

            data[:, :, :3] *= constants.l_aa2au

            self.assertTrue(np.allclose(
                data,
                np.genfromtxt(self.dir + '/data_' + _i).reshape(_n),
                atol=0.0
                ))

        # Some Negatives
        with self.assertRaises(ValueError):
            data = cpmd.cpmdReader(self.dir + '/MOMENTS_broken',
                                   filetype='MOMENTS',
                                   symbols=['X']*288)['data']
            data = cpmd.cpmdReader(self.dir + '/MOMENTS',
                                   filetype='MOMENTS',
                                   symbols=['X']*286)['data']
        # Test range
        data = cpmd.cpmdReader(self.dir + '/' + _i,
                               filetype='TRAJECTORY',
                               symbols=['X']*_n[1],
                               range=(2, 3, 6),
                               )['data']
        data[:, :, :3] *= constants.l_aa2au
        self.assertTrue(np.allclose(
            data,
            np.genfromtxt(self.dir + '/data_TRAJECTORY').reshape(_n)[2:8:3],
            atol=0
            ))

    def test_cpmdWriter(self):
        data_r = cpmd.cpmdReader(self.dir + '/TRAJECTORY',
                                 filetype='TRAJECTORY',
                                 symbols=['X']*208)['data']

        _outfile = 'OUT_cpmd_w'
        # --- important in case writer manipulates input
        data = copy.deepcopy(data_r)
        with self.assertRaises(ValueError):
            # --- sorted data
            cpmd.cpmdWriter(_outfile, data, symbols=['X', 'Y']*104,
                            write_atoms=True)
        cpmd.cpmdWriter(_outfile, data,
                        frames=np.arange(data.shape[0]).astype(int)+1,
                        write_atoms=False)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            data2 = cpmd.cpmdReader(_outfile,
                                    filetype='TRAJECTORY',
                                    symbols=cpmd.cpmd_kinds_from_file(_outfile)
                                    )['data']
        self.assertTrue(np.allclose(data_r, data2, atol=0.0))
        self.assertTrue(filecmp.cmp(_outfile,
                                    self.dir + '/TRAJECTORY',
                                    shallow=False),
                        f'CPMD file {self.dir}/TRAJECTORY reproduced '
                        f'incorrectly in {_outfile}'
                        )
        os.remove(_outfile)

        # --- selection
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            cpmd.cpmdWriter(_outfile, data,
                            frames=np.arange(data.shape[0]).astype(int)+1,
                            selection=[4, 9, 1, 33],
                            symbols=['X']*208,
                            pp='any',
                            write_atoms=True)
            data2 = cpmd.cpmdReader(_outfile,
                                    filetype='TRAJECTORY',
                                    symbols=['X']*4,
                                    )['data']
        self.assertTrue(np.allclose(data_r[:, [4, 9, 1, 33]], data2))

        os.remove(_outfile)
        os.remove(_outfile + '_ATOMS')
        os.remove(_outfile + '_ATOMS.xyz')

    def test_cpmdjob(self):
        # --- insufficiently tested
        #     needs also test for correct append behaviour
        for fn in ['cpmd_job_1.inp', 'cpmd_job_2.inp', 'cpmd_job_3.inp']:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ChirPyWarning)
                warnings.filterwarnings('ignore', category=FutureWarning)
                _cpmd = cpmd.CPMDjob.read_input_file(self.dir + '/' + fn)
                _cpmd.write_input_file("test.inp", fmt='angstrom')
                self.assertTrue(filecmp.cmp("test.inp",
                                            self.dir + '/' + fn,
                                            shallow=False),
                                'CPMDjob does not reproduce reference file: %s'
                                ' (see test.inp)'
                                % fn
                                )
                data = cpmd.cpmdReader(self.dir + '/TRAJECTORY',
                                       filetype='TRAJECTORY',
                                       symbols=['X']*208)
                _cpmd.ATOMS = _cpmd.ATOMS.from_data(data['symbols'][:6],
                                                    data['data'][0, :6, :3],
                                                    pp=_cpmd.ATOMS.pp)
                _cpmd.write_input_file("test.inp", fmt='angstrom')
                os.remove("test.inp")


class TestTinker(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_tinkermomentsReader(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            data = np.array(list(tinker.tinkermomentsReader(
                    *[self.dir + '/' + _f for _f in
                        ('s_0881.dip', 's_0881.magdip', 's_0881.ddip')],
                    columns='imddd'
                    )))
            self.assertTupleEqual(data.shape, (3, 16, 12))
            self.assertListEqual(
                    data[-1, 3].tolist(),
                    np.loadtxt(self.dir + '/data_s_0881_1').tolist()
                    )

            # -- wrong columns and other range
            data = np.array(list(tinker.tinkermomentsReader(
                    *[self.dir + '/' + _f for _f in
                        ('s_0881.dip', 's_0881.magdip', 's_0881.ddip')],
                    columns='iddd',
                    range=(0, 2, -1)
                    )))
            self.assertTupleEqual(data.shape, (2, 16, 12))
            self.assertListEqual(
                    data[-1, 3].tolist(),
                    np.loadtxt(self.dir + '/data_s_0881_2').tolist()
                    )

        # Some Negatives
            with self.assertRaises(ValueError):
                data = np.array(list(tinker.tinkermomentsReader(
                      *[self.dir + '/' + _f for _f in
                        ('s_0881_broken.dip', 's_0881.magdip', 's_0881.ddip')],
                      columns='imddd'
                      )))
