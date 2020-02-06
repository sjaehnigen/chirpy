#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------

import unittest
import os
import warnings
import filecmp
import numpy as np

from ..classes import system, quantum

# volume, trajectory, field, domain, core

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


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
        system = quantum.TimeDependentElectronDensity(fn, fn1, fn2, fn3)
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
