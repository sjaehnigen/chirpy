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

from chirpy.interface import cpmd, tinker, cp2k, orca, molden, gaussian
from chirpy import constants
from chirpy.config import ChirPyWarning

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_files'


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

    def test_cpmdReader_MOMENTS_position_form(self):
        # --- position form (d_au) must be auto-detected from column count
        #     (13 columns incl. frame index instead of 10) and correctly
        #     appended as the last 3 (of 12) data columns
        data = cpmd.cpmdReader(self.dir + '/MOMENTS_PF',
                               filetype='MOMENTS',
                               symbols=['X', 'X'])['data']

        self.assertTupleEqual(data.shape, (1, 2, 12))
        self.assertTrue(np.allclose(
            data[0, :, 9:12],
            [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]],
            ))
        # --- plain MOMENTS (9 data columns) has no position form
        data = cpmd.cpmdReader(self.dir + '/MOMENTS',
                               filetype='MOMENTS',
                               symbols=['X']*288)['data']
        self.assertEqual(data.shape[-1], 9)

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
            self.assertTrue(np.allclose(
                    data[-1, 3],
                    np.loadtxt(self.dir + '/data_s_0881_1'),
                    ))


            # -- wrong columns and other range
            data = np.array(list(tinker.tinkermomentsReader(
                    *[self.dir + '/' + _f for _f in
                        ('s_0881.dip', 's_0881.magdip', 's_0881.ddip')],
                    columns='iddd',
                    range=(0, 2, -1)
                    )))
            self.assertTupleEqual(data.shape, (2, 16, 12))
            self.assertTrue(np.allclose(
                    data[-1, 3],
                    np.loadtxt(self.dir + '/data_s_0881_2'),
                    ))

        # Some Negatives
            with self.assertRaises(ValueError):
                data = np.array(list(tinker.tinkermomentsReader(
                      *[self.dir + '/' + _f for _f in
                        ('s_0881_broken.dip', 's_0881.magdip', 's_0881.ddip')],
                      columns='imddd'
                      )))


class TestCP2K(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_parse_restart_file(self):
        _d = cp2k.parse_restart_file(self.dir + '/test.restart')

        self.assertIsInstance(_d, dict)
        self.assertIn('GLOBAL', _d)
        self.assertIn('FORCE_EVAL', _d)
        self.assertListEqual(_d['GLOBAL']['KEYWORDS'],
                             ['PROJECT test', 'RUN_TYPE MD'])

        # --- nested sections are parsed recursively
        _subsys = _d['FORCE_EVAL']['SUBSYS']
        self.assertIn('CELL', _subsys)
        self.assertIn('COORD', _subsys)
        self.assertListEqual(_subsys['CELL']['KEYWORDS'],
                             ['A 10.0 0.0 0.0',
                              'B 0.0 10.0 0.0',
                              'C 0.0 0.0 10.0'])
        self.assertListEqual(_subsys['COORD']['KEYWORDS'],
                             ['O 0.0 0.0 0.0', 'H 1.0 0.0 0.0'])


class TestOrca(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_orcaReader(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            data = orca.orcaReader(self.dir + '/test.hess')

        self.assertListEqual(data['symbols'], ['C', 'O'])
        self.assertTrue(np.allclose(
                            data['omega_cgs'],
                            [0., 0., 0., 0., 0., 1500.123456],
                            ))
        self.assertTrue(np.allclose(
                            data['pos_aa'],
                            [[0., 0., 0.],
                             [0., 0., 2.2 * constants.l_au2aa]],
                            ))
        self.assertTupleEqual(data['modes'].shape, (6, 2, 3))
        self.assertTrue(np.allclose(
                            data['modes'][5],
                            [[0.1, 0., 0.], [-0.1, 0., 0.]],
                            ))
        self.assertTupleEqual(data['APT_au'].shape, (2, 3, 3))

        # --- unknown/unsupported extensions raise
        with self.assertRaises(NotImplementedError):
            orca.orcaReader(self.dir + '/test.hess.unsupported')

    def test_orcaReader_invalid_format(self):
        with self.assertRaises(ValueError):
            orca.orcaReader(self.dir + '/test_bad.hess')


class TestMolden(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_moldenvib_roundtrip(self):
        symbols = ['O', 'H', 'H']
        coords_aa = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.96],
                              [0.9, 0.0, -0.3]])
        freqs = np.array([0., 0., 0., 0., 0., 0., 1600., 3650., 3750.])
        modes = np.random.RandomState(42).rand(9, 3, 3)

        _outfile = 'out.molden'
        molden.write_moldenvib_file(_outfile, symbols, coords_aa, freqs,
                                    modes)
        symbols2, coords2, freqs2, modes2 = \
            molden.read_moldenvib_file(_outfile)

        self.assertListEqual(symbols2, symbols)
        self.assertTrue(np.allclose(coords2, coords_aa, atol=1e-5))
        self.assertTrue(np.allclose(freqs2, freqs, atol=1e-5))
        self.assertTrue(np.allclose(modes2, modes, atol=1e-5))
        os.remove(_outfile)

    def test_read_moldenvib_file_wrong_format(self):
        _bad = 'bad.molden'
        with open(_bad, 'w') as _f:
            _f.write('not a molden file\n')
        with self.assertRaises(ValueError):
            molden.read_moldenvib_file(_bad)
        os.remove(_bad)


class TestGaussianNormalModes(unittest.TestCase):
    '''Test the internal (Gaussian-style) normal-mode calculation from a
       synthetic, translation-/rotation-invariant valence-force-field
       Hessian (built from pairwise-distance-only bond potentials, whose
       Hessian is exactly invariant under global translation/rotation).
       '''

    def setUp(self):
        self.masses = np.array([16.0, 1.0, 1.0])  # bent triatomic (H2O-like)
        self.coords = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 1.4, 1.1],
                [0.0, -1.4, 1.1],
                ])
        self.n_atoms = 3

        def _bond_hessian_block(k, ri, rj):
            d = ri - rj
            r = np.linalg.norm(d)
            u = d / r
            return k * np.outer(u, u)

        dim = self.n_atoms * 3
        hessian = np.zeros((dim, dim))
        for i, j, k in [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 0.3)]:
            h = _bond_hessian_block(k, self.coords[i], self.coords[j])
            hessian[3*i:3*i+3, 3*i:3*i+3] += h
            hessian[3*j:3*j+3, 3*j:3*j+3] += h
            hessian[3*i:3*i+3, 3*j:3*j+3] -= h
            hessian[3*j:3*j+3, 3*i:3*i+3] -= h
        self.hessian = hessian

    def test_calculate_normal_modes(self):
        e_vec, e_val, mwe_vec, cmc = gaussian.calculate_normal_modes(
                self.n_atoms, self.masses, self.coords, self.hessian)

        n_dof = 3 * self.n_atoms

        # --- 3N-6 vibrational modes remain after removing translation and
        #     rotation (non-linear molecule)
        self.assertEqual(len(e_val), n_dof - 6)
        self.assertEqual(e_vec.shape, (n_dof - 6, n_dof))
        self.assertEqual(mwe_vec.shape, (n_dof - 6, n_dof))

        # --- frequencies are real, sorted ascending, and positive (stable
        #     equilibrium geometry)
        self.assertTrue(np.all(np.isreal(e_val)))
        self.assertTrue(np.all(np.diff(e_val) >= -1e-8))
        self.assertTrue(np.all(np.array(e_val) > 0))

        # --- Cartesian (mass-weighted-space) eigenvectors are orthonormal
        self.assertTrue(np.allclose(
                            np.inner(e_vec, e_vec),
                            np.identity(n_dof - 6),
                            atol=1e-8,
                            ))

        # --- deterministic / reproducible
        e_vec2, e_val2, mwe_vec2, cmc2 = gaussian.calculate_normal_modes(
                self.n_atoms, self.masses, self.coords, self.hessian)
        self.assertTrue(np.allclose(e_val, e_val2))
        self.assertTrue(np.allclose(e_vec, e_vec2))

        # --- centre of mass w.r.t. coords (cmc = coords - COM) is returned
        self.assertTupleEqual(cmc.shape, self.coords.shape)

    def test_calculate_normal_modes_linear_molecule_limitation(self):
        # --- known ChirPy limitation: a diatomic (linear) molecule has only
        #     3N-5=1 vibrational mode, but the projector always removes a
        #     fixed set of 3 rotational directions, so for a 2-atom system
        #     (3N=6) all 6 modes get projected out, leaving none.
        masses = np.array([12.0, 16.0])
        coords = np.array([[0., 0., 0.], [0., 0., 2.2]])
        dim = 6
        hessian = np.zeros((dim, dim))
        hessian[2, 2] = 1.0
        hessian[5, 5] = 1.0
        hessian[2, 5] = hessian[5, 2] = -1.0

        e_vec, e_val, mwe_vec, cmc = gaussian.calculate_normal_modes(
                len(masses), masses, coords, hessian)
        self.assertEqual(len(e_val), 0)

    def test_diagonalize_dynamical_matrix(self):
        e_vec, e_val, mwevec = gaussian.diagonalize_dynamical_matrix(
                self.n_atoms, self.masses, self.hessian)

        n_dof = 3 * self.n_atoms
        self.assertEqual(e_vec.shape, (n_dof, n_dof))
        self.assertEqual(len(e_val), n_dof)
        # --- 6 zero-frequency modes for translation/rotation of a
        #     non-linear molecule
        self.assertEqual(np.sum(np.isclose(e_val, 0.0, atol=1e-4)), 6)
