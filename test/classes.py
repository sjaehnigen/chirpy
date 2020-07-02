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
import warnings
import filecmp
import numpy as np

from ..classes import system, quantum, trajectory

# volume, field, domain, core

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestTrajectory(unittest.TestCase):
    # --- insufficiently tested

    def setUp(self):
        self.dir = _test_dir + '/classes'

    def tearDown(self):
        pass

    def test_split(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            traj_6 = trajectory.XYZTrajectory.load(self.dir + '/ALANINE_NVT_6')
            traj_3 = trajectory.XYZTrajectory.load(self.dir + '/ALANINE_NVT_3')
        traj_6.split([4, 4, 0, 0, 0, 4], select=4)
        self.assertTrue(traj_3._is_similar(traj_6)[0] == 1)
        self.assertTrue(np.allclose(traj_3.data, traj_6.data))

    def test_iterator(self):
        traj = trajectory.XYZ(self.dir + '/traj_w_doubles.xyz',
                              range=(0, 10, 1000)
                              )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            traj.mask_duplicate_frames(verbose=False)
            ref = trajectory.XYZTrajectory.load(self.dir + '/TRAJ_clean')
        self.assertFalse(traj._is_equal(ref)[0] == 1)
        self.assertFalse(traj._is_equal(ref)[1][0] == 1)
        traj_e = traj.expand()
        self.assertTrue(traj_e._is_similar(ref)[0] == 1)
        self.assertTrue(np.allclose(traj_e.data, ref.data))

        self.assertTrue(len(traj.expand().data), 0)
        traj.rewind()
        traj_e = traj.expand()
        self.assertTrue(traj_e._is_similar(ref)[0] == 1)
        self.assertTrue(np.allclose(traj_e.data, ref.data))


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
            warnings.filterwarnings('ignore', category=UserWarning)
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
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            system.rho.aim(verbose=False)
        system.calculate_velocity_field(lower_thresh=thresh)
        system.v.helmholtz_decomposition()
        self.assertTrue(np.allclose(system.v.data,
                                    system.v.solenoidal_field +
                                    system.v.irrotational_field,
                                    atol=thresh
                                    ))

        system.rho.sparsity(2)
        system.j.sparsity(2)
        system.rho.write(self.dir + "/out.cube")
        os.remove(self.dir + "/out.cube")
