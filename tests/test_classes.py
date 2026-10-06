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
import warnings
import filecmp
import numpy as np

from chirpy.classes import system, quantum, trajectory, core, volume
from chirpy.config import ChirPyWarning
from chirpy import constants
from chirpy.physics import statistical_mechanics

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_files'


def _func(x0, x1):
    # --- some example array manipulation
    r0 = x0 + x1.swapaxes(1, 2)
    r0 = np.linalg.norm(r0, axis=-2)
    return r0.T


class TestCore(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_palarray(self):
        d0 = np.random.rand(8, 8, 8, 13)
        d1 = np.random.rand(8, 8, 8, 17)
        JOB = core.PALARRAY(_func, d0, repeat=2, axis=3, n_cores=1)
        S = JOB.run()
        r0 = d0[:, :, :, :, None] + d0[:, :, :, None, :].swapaxes(1, 2)
        r0 = np.linalg.norm(r0, axis=1).swapaxes(0, 1)
        r0 = np.moveaxis(r0, -1, 0)
        r0 = np.moveaxis(r0, -1, 0)
        self.assertTrue(np.allclose(S, r0))
        JOB = core.PALARRAY(_func, d0, repeat=2, upper_triangle=True, axis=3)
        S = JOB.run()
        r0[np.tril_indices(13, -1)] = 0.0
        self.assertTrue(np.allclose(S, r0))

        JOB = core.PALARRAY(_func, d0, d1, axis=3)
        S = JOB.run()
        r0 = d0[:, :, :, :, None] + d1[:, :, :, None, :].swapaxes(1, 2)
        r0 = np.linalg.norm(r0, axis=1).swapaxes(0, 1)
        r0 = np.moveaxis(r0, -1, 0)
        r0 = np.moveaxis(r0, -1, 0)
        self.assertTrue(np.allclose(S, r0))


class TestTrajectory(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_split(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            traj_6 = trajectory._XYZTrajectory.load(self.dir+'/ALANINE_NVT_6')
            traj_3 = trajectory._XYZTrajectory.load(self.dir+'/ALANINE_NVT_3')
        traj_6.split([4, 4, 0, 0, 0, 4], select=4)
        self.assertTrue(traj_3._is_similar(traj_6)[0] == 1)
        self.assertTrue(np.allclose(traj_3.data, traj_6.data))

    def test_iterator(self):
        traj = trajectory.XYZ(self.dir + '/traj_w_doubles.xyz',
                              range=(0, 10, 1000)
                              )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            traj.mask_duplicate_frames(verbose=False)
            ref = trajectory._XYZTrajectory.load(self.dir + '/TRAJ_clean')
        self.assertFalse(traj._is_equal(ref)[0] == 1)
        self.assertFalse(traj._is_equal(ref)[1][0] == 1)
        traj_e = traj.expand()
        self.assertTrue(traj_e._is_similar(ref)[0] == 1)
        self.assertListEqual(traj_e.data.tolist(), ref.data.tolist())

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            self.assertTrue(traj.expand() is None)
        traj.rewind()
        traj_e = traj.expand()
        self.assertTrue(traj_e._is_similar(ref)[0] == 1)
        self.assertTrue(np.allclose(traj_e.data, ref.data))

    def test_expand_batch(self):
        # --- reading the full trajectory in fixed-size batches should
        #     reproduce the same data as reading it in one go
        full = trajectory.XYZ(self.dir + '/trajectory.xyz').expand()
        n_frames = full.data.shape[0]

        traj = trajectory.XYZ(self.dir + '/trajectory.xyz')
        batches = []
        while True:
            _b = traj.expand(batch=4, ignore_warning=True)
            if _b is None:
                break
            batches.append(_b.data)

        # --- batches of size 4 over 10 frames: 4, 4, 2
        self.assertListEqual([_b.shape[0] for _b in batches], [4, 4, 2])
        self.assertTrue(np.allclose(
                            np.concatenate(batches, axis=0),
                            full.data,
                            ))
        self.assertEqual(sum(_b.shape[0] for _b in batches), n_frames)

        # --- an exhausted iterator returns None (with a warning) instead
        #     of raising
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertIsNone(traj.expand(batch=4))
            self.assertTrue(any('exhausted' in str(_w.message) for _w in w))

    def test_alignment(self):
        _ref = self.dir + '/trajectory_aligned.xyz'
        _out = 'out_align.xyz'
        traj = trajectory.XYZ(self.dir + '/trajectory.xyz')
        traj.align_coordinates()
        traj_ref = trajectory.XYZ(_ref)
        self.assertTrue(np.allclose(traj.data, traj_ref.data))

        # traj = trajectory._XYZTrajectory(self.dir + '/trajectory.xyz')
        traj = trajectory.XYZ(self.dir + '/trajectory.xyz').expand()
        traj.align_coordinates(selection=list(range(12)))
        traj.write(_out)
        self.assertTrue(
               filecmp.cmp(_out, _ref, shallow=False),
               f'Trajectory reproduced incorrectly: {_ref})'
               )
        os.remove(_out)

    def test_centering(self):
        _ref = self.dir + '/trajectory_centered_at_O.xyz'
        _out = 'out_center.xyz'
        traj = trajectory.XYZ(self.dir + '/trajectory.xyz')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            traj.center_coordinates()
        traj_ref = trajectory.XYZ(_ref)
        self.assertTrue(np.allclose(traj.data, traj_ref.data))

        traj = trajectory.XYZ(self.dir + '/trajectory.xyz').expand()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            traj.center_coordinates(selection=list(range(12)))
        traj.write(_out)
        self.assertTrue(
               filecmp.cmp(_out, _ref, shallow=False),
               f'Trajectory reproduced incorrectly: {_ref})'
               )
        os.remove(_out)

    def test_clean_velocities(self):
        _ref = self.dir + '/trajectory_cleaned.xyz'
        _out = 'out_clean.xyz'
        traj = trajectory.XYZ(self.dir + '/trajectory.xyz')
        traj.clean_velocities()
        traj_ref = trajectory.XYZ(_ref)
        self.assertTrue(np.allclose(traj.data, traj_ref.data))

        traj = trajectory.XYZ(self.dir + '/trajectory.xyz').expand()
        traj.clean_velocities()
        traj.write(_out)
        self.assertTrue(
               filecmp.cmp(_out, _ref, shallow=False),
               f'Trajectory reproduced incorrectly: {_ref})'
               )
        os.remove(_out)

    def test_center_of_weight_with_mask(self):
        _fn = self.dir + '/two_waters.xyz'
        masses = np.array([15.999, 1.008, 1.008])
        pos0 = np.array([[0., 0., 0.],
                         [0.958, 0., 0.],
                         [-0.240, 0.927, 0.]])
        ref_com0 = (pos0 * masses[:, None]).sum(axis=0) / masses.sum()
        ref_cog0 = pos0.mean(axis=0)
        mask = [0, 0, 0, 1, 1, 1]

        traj = trajectory.XYZ(_fn)
        traj.center_of_mass(mask=mask)
        self.assertTupleEqual(traj.mol_com_aa.shape, (2, 3))
        self.assertTrue(np.allclose(traj.mol_com_aa[0], ref_com0))

        # --- center_of_geometry (unweighted) with mask+join_molecules
        # (previously crashed with AttributeError, cf. _center_of_weight)
        traj_g = trajectory.XYZ(_fn)
        traj_g.center_of_geometry(mask=mask)
        self.assertTupleEqual(traj_g.mol_cog_aa.shape, (2, 3))
        self.assertTrue(np.allclose(traj_g.mol_cog_aa[0], ref_cog0))

        # --- without molecule joining, cowt is computed directly (no
        # wrap_molecules side effect required)
        traj_nj = trajectory.XYZ(_fn)
        traj_nj.center_of_mass(mask=mask, join_molecules=False)
        self.assertTrue(np.allclose(traj_nj.mol_com_aa[0], ref_com0))

    def test_wrap(self):
        cell = np.array([10., 10., 10., 90., 90., 90.])
        pos = np.array([[11.0, -2.0, 5.0],
                        [3.0, 3.0, 3.0]])

        traj = trajectory._XYZTrajectory(data=pos[None],
                                         symbols=['H', 'H'],
                                         cell_aa_deg=cell)
        traj.wrap()
        self.assertTrue(np.allclose(traj.pos_aa[0], [[1.0, 8.0, 5.0],
                                                      [3.0, 3.0, 3.0]]))

        # --- _MOMENTS.wrap() follows the same convention for positions
        data = np.zeros((1, 2, 12))
        data[0, :, :3] = pos
        mom = trajectory._MOMENTSTrajectory(data=data,
                                            symbols=['X', 'X'],
                                            cell_aa_deg=cell)
        mom.wrap()
        self.assertTrue(np.allclose(mom.pos_aa[0], [[1.0, 8.0, 5.0],
                                                     [3.0, 3.0, 3.0]]))

    def test_wrap_molecules(self):
        cell = np.array([10., 10., 10., 90., 90., 90.])
        pos = np.array([[0.5, 0.5, 0.5],
                        [1.458, 0.5, 0.5],
                        [0.26, 1.427, 0.5],
                        [0.2, 5.0, 5.0],
                        [9.8, 5.0, 5.0],
                        [0.958, 5.927, 5.0]])
        symbols = ['O', 'H', 'H', 'O', 'H', 'H']
        mol_map = [0, 0, 0, 1, 1, 1]

        traj = trajectory._XYZTrajectory(data=pos[None],
                                         symbols=symbols,
                                         cell_aa_deg=cell)
        # --- a plain wrap() does not restore molecular connectivity
        # across periodic boundaries: atom 4 stays far from its molecule
        traj_plain = trajectory._XYZTrajectory(data=pos[None],
                                               symbols=symbols,
                                               cell_aa_deg=cell)
        traj_plain.wrap()
        self.assertGreater(
                np.linalg.norm(traj_plain.pos_aa[0, 3]
                               - traj_plain.pos_aa[0, 4]),
                5.0
                )

        # --- wrap_molecules() keeps molecules whole (correct periodic
        # image chosen relative to the reference atom of each molecule)
        traj.wrap_molecules(mol_map)
        self.assertTrue(np.allclose(traj.pos_aa[0, 4], [-0.2, 5.0, 5.0]))
        self.assertLess(
                np.linalg.norm(traj.pos_aa[0, 3] - traj.pos_aa[0, 4]),
                1.0
                )


class TestSystem(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_supercell(self):
        largs = {
                'fn_topo': self.dir + "/topo.pdb",
                'range': (0, 3, 10),
                'sort': True,
                }
        _load = system.Supercell(self.dir + "/MD-NVT-production-pos-1.xyz",
                                 fmt='xyz', **largs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            skip = _load.XYZ.mask_duplicate_frames(verbose=False)
        largs.update({'skip': skip})

        nargs = {}
        for _a in [
            'range',
            'fn_topo',
            'sort',
            'skip',
                   ]:
            nargs[_a] = largs.get(_a)

        _load_vel = system.Supercell(self.dir + "/MD-NVT-production-vel-1.xyz",
                                     fmt='xyz', **nargs)
        _load.XYZ.merge(_load_vel.XYZ, axis=-1)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load.extract_molecules([10, 11])
        _load.write("out_system.xyz", fmt='xyz', rewind=False)

        self.assertTrue(filecmp.cmp("out_system.xyz",
                                    self.dir + "/ref.xyz",
                                    shallow=False),
                        'Class does not reproduce reference '
                        f'{self.dir}/ref.xyz!',
                        )
        os.remove("out_system.xyz")

        # --- selection
        _load = system.Supercell(self.dir + "/MD-NVT-production-pos-1.xyz",
                                 self.dir + "/MD-NVT-production-vel-1.xyz",
                                 **largs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            skip = _load.XYZ.mask_duplicate_frames(verbose=False)

        sel = [_i
               for _m in [10, 11]
               for _i, _s in enumerate(_load.mol_map)
               if _s == _m]

        _load.write("out_system.xyz", fmt='xyz', selection=sel, rewind=True)

        self.assertTrue(filecmp.cmp("out_system.xyz",
                                    self.dir + "/ref.xyz",
                                    shallow=False),
                        'Class does not reproduce reference '
                        f'{self.dir}/ref.xyz!',
                        )
        os.remove("out_system.xyz")

    def test_sort(self):
        _load = system.Supercell(self.dir + '/input_sort.xyz', sort=False)
        ref_symbols = tuple(sorted(_load.XYZ.symbols))
        _MF = _load.molecular_formula
        self.assertTupleEqual(
                ref_symbols,
                tuple([_is for _s in _MF for _is in _MF[_s]*(_s,)]),
                'Got wrong or unsorted molecular formula'
                )
        _load = system.Supercell(self.dir + '/input_sort.xyz', sort=True)
        self.assertTupleEqual(
                ref_symbols,
                _load.XYZ.symbols,
                'system symbols not sorted'
                )
        self.assertTupleEqual(
                _load.XYZ.symbols,
                _load.symbols,
                'system sort inconsistent'
                )

    def test_repeat(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load = system.Supercell(self.dir + '/input_sort.xyz', sort=False)

        n_atoms = len(_load.XYZ.symbols)
        _load.repeat(2)

        # --- 2x2x2 = 8 -fold duplication of atoms and symbols
        self.assertEqual(len(_load.XYZ.symbols), 8 * n_atoms)
        self.assertEqual(len(_load.symbols), 8 * n_atoms)
        self.assertTupleEqual(_load.XYZ.pos_aa.shape, (8 * n_atoms, 3))

        # --- non-cubic repeat: only duplicate along x (fresh instance,
        # since system._copy() is a shallow [BETA] copy not suited for
        # independent mutation of the same loaded system)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load_x = system.Supercell(self.dir + '/input_sort.xyz',
                                       sort=False)
        _load_x.repeat((2, 1, 1))
        self.assertEqual(len(_load_x.XYZ.symbols), 2 * n_atoms)

    def test_add(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load1 = system.Supercell(self.dir + '/input_sort.xyz',
                                      sort=False)
            _load2 = system.Supercell(self.dir + '/input_sort.xyz',
                                      sort=False)

        n_atoms = len(_load1.XYZ.symbols)
        _added = _load1 + _load2

        self.assertEqual(len(_added.symbols), 2 * n_atoms)
        self.assertTupleEqual(_added.symbols[:n_atoms], _load1.symbols)
        self.assertTupleEqual(_added.symbols[n_atoms:], _load2.symbols)
        self.assertTrue(np.allclose(_added.XYZ.pos_aa[:n_atoms],
                                    _load1.XYZ.pos_aa))
        self.assertTrue(np.allclose(_added.XYZ.pos_aa[n_atoms:],
                                    _load2.XYZ.pos_aa))

    def test_extract_atoms(self):
        largs = {
                'fn_topo': self.dir + "/topo.pdb",
                'range': (0, 3, 10),
                'sort': True,
                }
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load = system.Supercell(
                    self.dir + "/MD-NVT-production-pos-1.xyz",
                    fmt='xyz', **largs)

        mol_map = np.array(_load.mol_map)
        sel = [_i for _i, _m in enumerate(mol_map) if _m in (10, 11)]

        _load.extract_atoms(sel)

        # --- equivalent to selecting the same atoms via extract_molecules
        self.assertEqual(len(_load.symbols), len(sel))
        self.assertTupleEqual(_load.XYZ.pos_aa.shape, (len(sel), 3))
        # --- mol_map must be re-indexed consistently with the selection
        # (previously used membership test on wrong index domain, cf.
        # extract_atoms)
        self.assertTrue(np.array_equal(_load.mol_map, mol_map[sel]))
        self.assertTrue(set(_load.mol_map).issubset({10, 11}))

    def test_split_pdb_by_atom_names(self):
        # --- splitting a trajectory loaded with a pdb topology using the
        # atom names taken from that topology (instead of a mol_map)
        largs = {
                'fn_topo': self.dir + "/topo.pdb",
                'range': (0, 3, 10),
                'sort': True,
                }
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            _load = system.Supercell(
                    self.dir + "/MD-NVT-production-pos-1.xyz",
                    fmt='xyz', **largs)

        names = _load.XYZ.names
        n_atoms = len(_load.XYZ.symbols)
        n_carbon = sum(1 for _n in names if _n == 'C')
        self.assertGreater(n_carbon, 0)

        _load.XYZ.split(names, select=['C'])

        self.assertEqual(len(_load.XYZ.symbols), n_carbon)
        self.assertLess(len(_load.XYZ.symbols), n_atoms)
        self.assertSetEqual(set(_load.XYZ.names), {'C'})
        self.assertSetEqual(set(_load.XYZ.symbols), {'C'})


class TestQuantum(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_electronic_system(self):

        fn = self.dir + "/DENSITY-000001-SPARSE.cube"
        fn1 = self.dir + "/CURRENT-000001-1-SPARSE.cube"
        fn2 = self.dir + "/CURRENT-000001-2-SPARSE.cube"
        fn3 = self.dir + "/CURRENT-000001-3-SPARSE.cube"

        thresh = 5.E-3
        system = quantum.TDElectronDensity(fn, fn1, fn2, fn3)
        system.auto_crop(thresh=thresh)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            system.rho.aim(verbose=False)
        system.calculate_velocity_field(thresh=thresh)
        system.v.helmholtz_decomposition()
        self.assertTrue(np.allclose(system.v.data,
                                    system.v.solenoidal_field.data +
                                    system.v.irrotational_field.data,
                                    atol=thresh
                                    ))

        system.rho = system.rho.sparse(2)
        system.j = system.j.sparse(2)
        system.rho.write(self.dir + "/out.cube")
        os.remove(self.dir + "/out.cube")


class TestVibrationalModes(unittest.TestCase):
    '''VibrationalModes/_MODES was previously completely untested (no test
       file referenced these classes at all).'''

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_load_and_orthonormality_check(self):
        # --- the shipped fixture is a real-world xvibs file whose
        #     eigenvectors are known to be not perfectly orthonormal;
        #     loading it must succeed and the built-in orthonormality
        #     check must warn about this instead of silently ignoring it
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            vib = trajectory.VibrationalModes(self.dir + '/test.xvibs')
            _msgs = [str(_w.message) for _w in w]

        self.assertEqual(vib.n_modes, 3 * vib.n_atoms)
        self.assertEqual(vib.eival_cgs.shape, (vib.n_modes, ))
        self.assertEqual(vib.eivec.shape, (vib.n_modes, vib.n_atoms, 3))
        self.assertTrue(any('not orthonormal' in _m for _m in _msgs))

        # --- units: no IR intensities given --> defaults to one and warns
        self.assertTrue(any('IR intensities' in _m for _m in _msgs))
        self.assertTrue(np.allclose(vib.IR_kmpmol, 1.0))

    def test_select_modes(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib = trajectory.VibrationalModes(self.dir + '/test.xvibs')

        sub = vib.select_modes([6, 7, 8])
        self.assertEqual(sub.n_modes, 3)
        self.assertTrue(np.allclose(sub.data, vib.data[[6, 7, 8]]))
        # --- original object remains unaffected (deep copy)
        self.assertEqual(vib.n_modes, 39)

        # --- a single integer is also accepted
        single = vib.select_modes(6)
        self.assertEqual(single.n_modes, 1)

        with self.assertRaises(TypeError):
            vib.select_modes('6')

    def test_repeat(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib = trajectory.VibrationalModes(self.dir + '/test.xvibs')

        n_atoms = vib.n_atoms
        n_modes = vib.n_modes
        vib.repeat(2)

        # --- 2x2x2 = 8-fold duplication of atoms, mode count unchanged
        self.assertEqual(vib.n_atoms, 8 * n_atoms)
        self.assertEqual(vib.n_modes, n_modes)
        self.assertTupleEqual(vib.data.shape, (n_modes, 8 * n_atoms, 9))

    def test_calculate_nuclear_velocities(self):
        # --- equipartition theorem: velocities generated from normal modes
        # (occupation='single') must reproduce the requested temperature,
        # i.e. average kinetic energy per mode == 0.5 * k_B * T
        # (previously scaled velocities linearly by 'scale' instead of
        # sqrt(scale), overshooting the target temperature by a factor of
        # 'scale' since kinetic energy is quadratic in velocity)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib = trajectory.VibrationalModes(self.dir + '/test.xvibs')

        T = 300.
        vib.calculate_nuclear_velocities(occupation='single', temperature=T)

        self.assertTupleEqual(vib.vel_au.shape, (vib.n_modes, vib.n_atoms, 3))

        e_kin_au = statistical_mechanics.kinetic_energies(vib.vel_au,
                                                          vib.masses_amu)
        T_eff = 2 * e_kin_au.sum(axis=1).mean() / constants.k_B_au
        self.assertAlmostEqual(T_eff, T, places=6)

    def test_rotate(self):
        # --- rotate() previously crashed for VibrationalModes/'modes'-type
        # objects with an UnboundLocalError (pos_aa/vel_au were referenced
        # unconditionally but never assigned in the 'modes' branch)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib = trajectory.VibrationalModes(self.dir + '/test.xvibs')

        theta = np.pi / 5
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

        pos0 = vib.pos_aa.copy()
        modes0 = vib.modes.copy()
        vib.rotate(R)

        self.assertTrue(np.allclose(vib.pos_aa,
                                    np.einsum('ji,fai->faj', R, pos0)))
        self.assertTrue(np.allclose(vib.modes,
                                    np.einsum('ji,fai->faj', R, modes0)))

        # --- rank-2 tensors (APT/AAT) must rotate as R @ T @ R^T per atom,
        # and derived (e/m)transition dipole moments must stay consistent
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib2 = trajectory.VibrationalModes(self.dir + '/test.xvibs')
        rng = np.random.default_rng(0)
        vib2.APT_au = rng.random((vib2.n_atoms, 3, 3))
        vib2.AAT_au = rng.random((vib2.n_atoms, 3, 3))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib2._sync_class(check_orthonormality=False)

        apt0 = vib2.APT_au.copy()
        aat0 = vib2.AAT_au.copy()
        etdm0 = vib2.etdm_au.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ChirPyWarning)
            vib2.rotate(R)

        self.assertTrue(np.allclose(
                vib2.APT_au, np.einsum('ij,ajk,lk->ail', R, apt0, R)))
        self.assertTrue(np.allclose(
                vib2.AAT_au, np.einsum('ij,ajk,lk->ail', R, aat0, R)))
        self.assertTrue(np.allclose(
                vib2.etdm_au, np.einsum('ij,aj->ai', R, etdm0)))


class TestVolume(unittest.TestCase):
    '''ScalarField/VectorField had no dedicated unit tests.'''

    def setUp(self):
        self.cell_vec_aa = np.diag([1.0, 1.0, 1.0])

    def tearDown(self):
        pass

    def test_crop(self):
        n = 10
        data = np.zeros((n, n, n))
        data[3:7, 3:7, 3:7] = 1.0

        sf = volume.ScalarField.from_data(data.copy(), self.cell_vec_aa)
        sf.crop([(2, 8), (2, 8), (2, 8)], dims='xyz')

        self.assertTupleEqual(sf.data.shape, (6, 6, 6))
        self.assertTrue(np.allclose(sf.data, data[2:8, 2:8, 2:8]))
        # --- origin shifts by the cropped-away lower bound
        self.assertTrue(np.allclose(sf.origin_aa, [2., 2., 2.]))

    def test_auto_crop(self):
        n = 10
        data = np.zeros((n, n, n))
        data[3:7, 3:7, 3:7] = 1.0

        sf = volume.ScalarField.from_data(data.copy(), self.cell_vec_aa)
        _r = sf.auto_crop(thresh=0.5, dry_run=True)
        self.assertEqual(_r, ((3, 6), (3, 6), (3, 6)))

        # --- data unaffected by dry_run
        self.assertTupleEqual(sf.data.shape, (n, n, n))

        sf.auto_crop(thresh=0.5)
        self.assertTupleEqual(sf.data.shape, (3, 3, 3))
        self.assertTrue(np.all(sf.data > 0.5))

    def test_normalise(self):
        n = 6
        data = np.full((n, n, n), 4.0)

        # --- normalise by a constant float divides all grid points by it
        sf = volume.ScalarField.from_data(data.copy(), self.cell_vec_aa)
        sf.normalise(norm=2.0)
        self.assertTrue(np.allclose(sf.data, 2.0))

        # --- with no norm given, a VectorField is normalised to unit
        #     vectors using np.linalg.norm along the given axis
        vx = np.full((n, n, n), 3.0)
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        vf = volume.VectorField.from_data(np.array([vx, vy, vz]),
                                          self.cell_vec_aa)
        vf.normalise(axis=0)
        self.assertTrue(np.allclose(vf.data[:, 0, 0, 0], [1., 0., 0.]))

        # --- grid points where the norm is below threshold are set to
        #     zero instead of causing a division by zero
        sf_zero = volume.ScalarField.from_data(
                np.full((n, n, n), 5.0), self.cell_vec_aa)
        sf_zero.normalise(norm=0.0, thresh=1e-8)
        self.assertTrue(np.allclose(sf_zero.data, 0.0))

    def test_add_with_different_grids(self):
        # --- add() interpolates the second field onto the first field's
        #     grid, so grids of disparate resolution/spacing can be summed
        data_a = np.zeros((10, 10, 10))
        data_a[5, 5, 5] = 1.0
        sf_a = volume.ScalarField.from_data(data_a, np.diag([1.0, 1.0, 1.0]))

        data_b = np.ones((20, 20, 20)) * 0.1
        sf_b = volume.ScalarField.from_data(data_b, np.diag([0.5, 0.5, 0.5]))

        sf_sum = sf_a + sf_b
        # --- keeps the grid/shape of the left-hand operand
        self.assertTupleEqual(sf_sum.data.shape, data_a.shape)
        self.assertAlmostEqual(sf_sum.data.min(), 0.1, places=6)
        self.assertAlmostEqual(sf_sum.data.max(), 1.1, places=6)

        # --- subtraction is the inverse operation
        sf_diff = sf_sum - sf_b
        self.assertTrue(np.allclose(sf_diff.data, sf_a.data, atol=1e-6))

    def test_streamlines(self):
        # --- uniform flow field along +x: a particle released anywhere
        #     should be advected in a straight line along x only
        n = 10
        vx = np.ones((n, n, n))
        vy = np.zeros((n, n, n))
        vz = np.zeros((n, n, n))
        vf = volume.VectorField.from_data(np.array([vx, vy, vz]),
                                          self.cell_vec_aa)

        start = np.array([[2.0, 5.0, 5.0]])
        out = vf.streamlines(start, sparse=1, length=10, timestep_fs=50.0,
                             forward=True, backward=False)

        _path = out['streamlines'][:, 0, :3]
        # --- x-coordinate increases monotonically
        self.assertTrue(np.all(np.diff(_path[:, 0]) > 0))
        # --- y/z stay fixed (flow is purely along x)
        self.assertTrue(np.allclose(_path[:, 1], 5.0))
        self.assertTrue(np.allclose(_path[:, 2], 5.0))
