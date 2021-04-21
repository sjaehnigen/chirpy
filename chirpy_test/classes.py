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
import warnings
import filecmp
import numpy as np

from chirpy.classes import system, quantum, trajectory, core
from chirpy.config import ChirPyWarning

# volume, field, domain

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


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

        self.assertTrue(traj.expand() is None)
        traj.rewind()
        traj_e = traj.expand()
        self.assertTrue(traj_e._is_similar(ref)[0] == 1)
        self.assertTrue(np.allclose(traj_e.data, ref.data))

    def test_alignment(self):
        _ref = self.dir + '/trajectory_aligned.xyz'
        _out = 'out_align.xyz'
        traj = trajectory.XYZ(self.dir + '/trajectory.xyz')
        traj.align_coordinates()
        traj_ref = trajectory.XYZ(_ref)
        self.assertTrue(np.allclose(traj.data, traj_ref.data))

        traj = trajectory._XYZTrajectory(self.dir + '/trajectory.xyz')
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

        traj = trajectory._XYZTrajectory(self.dir + '/trajectory.xyz')
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

        traj = trajectory._XYZTrajectory(self.dir + '/trajectory.xyz')
        traj.clean_velocities()
        traj.write(_out)
        self.assertTrue(
               filecmp.cmp(_out, _ref, shallow=False),
               f'Trajectory reproduced incorrectly: {_ref})'
               )
        os.remove(_out)


class TestSystem(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_supercell(self):
        largs = {
                'fn_topo': self.dir + "/topo.pdb",
                'range': (0, 7, 24),
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

        _load.extract_molecules([10, 11])
        _load.write(self.dir + "/out.xyz", fmt='xyz', rewind=False)

        self.assertTrue(filecmp.cmp(self.dir + "/out.xyz",
                                    self.dir + "/ref.xyz",
                                    shallow=False),
                        'Class does not reproduce reference file!',
                        )
        os.remove(self.dir + "/out.xyz")


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
