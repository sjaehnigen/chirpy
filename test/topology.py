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

import os
import unittest
import numpy as np
from functools import partial

from ..topology import mapping, dissection, motion, grid  # , distribution
from ..read import coordinates
from ..physics import constants

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/.test_files'


class TestMapping(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dist_crit_aa(self):
        symbols = tuple(constants.rvdw)[:-1]
        crit = mapping.dist_crit_aa(symbols)

        self.assertIsInstance(crit, np.ndarray)
        self.assertTupleEqual(crit.shape, (len(symbols), len(symbols)))
        self.assertEqual(len(set(np.around(
                          crit.diagonal() / constants.symbols_to_rvdw(symbols),
                          decimals=3
                          ))), 1)

    def test_dec(self):
        a = np.tile(np.arange(20), (3, 1)).T
        i = 10 * (1,) + 10 * (3,)
        dec = mapping.dec(a, i)

        self.assertIsInstance(dec, list)
        self.assertEqual(len(dec), max(i) + 1)
        self.assertTrue(np.allclose(
                       np.concatenate(
                         tuple([_d.flatten() for _d in dec])).reshape((20, 3)),
                       a
                       ))

    def test_cowt(self):
        a = np.array(
                [[
                        [1., 1., 0.32],
                        [2.3, -1., -9.],
                        [-0.23, 8.2, 0.],
                        [-1.23, 8.2, 0.]
                ], [
                        [4., -2, 0.32],
                        [0.3, 1., -8.],
                        [0.2, 1., -8.],
                        [-1.45, 2.2, 1.]
                ]]
                )
        cowt = mapping.cowt(a, wt=(1, 1, 3, 4))
        self.assertTupleEqual(cowt.shape, (2, 3))
        self.assertListEqual(np.around(cowt, decimals=3).flatten().tolist(),
                             [-0.257, 6.378, -0.964, -0.1, 1.2, -3.076]
                             )
        cowt_b = mapping.cowt(a[1], wt=(1, 1, 3, 4))
        self.assertTupleEqual(cowt_b.shape, (3,))
        self.assertListEqual(cowt[1].tolist(), cowt_b.tolist())

    def test_get_cell_vec(self):
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        cell_vec_aa = np.around(mapping.get_cell_vec(cell_aa_deg), decimals=3)
        self.assertListEqual(
                cell_vec_aa.tolist(),
                [[24.218, 0., 0.], [0., 15.92, 0.], [-4.995, 0., 12.393]]
                )
        cell_vec_aa = np.around(mapping.get_cell_vec(cell_aa_deg,
                                                     priority=(2, 0, 1)),
                                decimals=3)
        self.assertListEqual(
                cell_vec_aa.tolist(),
                [[22.462, 0., -9.053], [0., 15.92, 0.], [0., 0., 13.362]]
                )

    def test_detect_lattice(self):
        # --- Insufficiently tested
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        lattice = mapping.detect_lattice(cell_aa_deg)
        self.assertEqual(lattice, 'monoclinic')

    def test_wrap(self):
        cell_aa_deg = np.array([1., 2., np.sqrt(2), 90., 135., 90.])
        _p1 = np.ones((2, 3))
        _p1[0] = [0.99, 1., 0.5]
        _p1[1] = [1.1, 2.1, 1.1]
        _p1 = np.around(mapping.wrap(_p1, cell_aa_deg), decimals=3)
        self.assertListEqual(_p1.flatten().tolist(),
                             [-0.01, 1., 0.5, 0.1, 0.1, 0.1])
        _shape = (3, 2, 3)
        _p2 = np.ones(_shape)
        self.assertTupleEqual(mapping.wrap(_p2, cell_aa_deg).shape, _shape)

    def test_distance_pbc(self):
        cell_aa_deg = np.array([1., 2., np.sqrt(2), 90., 135., 90.])
        _d = np.around(mapping.distance_pbc(np.array([0.99, 1., 0.5]),
                                            np.array([1.1, 2.1, 1.1]),
                                            cell_aa_deg=cell_aa_deg),
                       decimals=3)
        self.assertListEqual(_d.tolist(), [0.11, -0.9, -0.4])

        _d = np.around(mapping.distance_pbc(np.array([0.15, 1.0, 0.5]),
                                            np.array([0.85, 1.0, 0.5]),
                                            cell_aa_deg=cell_aa_deg),
                       decimals=3)
        self.assertListEqual(_d.tolist(), [-0.3, 0., 0.])

    def test_distance_matrix(self):
        _p = np.ones((2, 3))
        _p[0] = [0.99, 1., 0.5]
        _p[1] = [1.1, 2.1, 1.1]
        dist = mapping.distance_matrix(_p, cartesian=True)
        self.assertTupleEqual(dist.shape, (2, 2, 3))

        dist = mapping.distance_matrix(_p)
        self.assertTupleEqual(dist.shape, (2, 2))
        self.assertListEqual(dist.diagonal().tolist(), 2 * [0.])

        dist2 = mapping.distance_matrix(_p, _p[::-1])
        self.assertListEqual(dist.tolist(), dist2[:, ::-1].tolist())


class TestDissection(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_fermi_cutoff_function(self):
        _f = partial(dissection.fermi_cutoff_function, R_cutoff=5, D=2.5)
        _r = np.around(list(map(_f, range(0, 10))), decimals=3)
        self.assertListEqual(
             _r.tolist(),
             [0.881, 0.832, 0.769, 0.69, 0.599, 0.5, 0.401, 0.31, 0.231, 0.168]
             )

    def test_define_molecules(self):
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        pos_aa, symbols, comments = coordinates.xyzReader(
                self.dir + '/test_long_mol.xyz')
        mol_map = dissection.define_molecules(pos_aa[0], symbols,
                                              cell_aa_deg=cell_aa_deg)
        ref_map = np.array([
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 9, 9, 10, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 11, 11, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
            4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 11, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 4, 4, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            11, 11, 11, 11, 11, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
            3, 3, 3, 4, 4, 4, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 9, 9, 9, 7, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 13, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 11, 11, 11, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 3, 16, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 10, 10, 8, 7, 14, 13, 12,
            12, 4, 4, 15, 16, 2, 1, 6, 6, 9, 9, 7, 8, 13, 14, 11, 11, 3, 3, 16,
            15, 1, 2
            ])
        _list = [tuple(np.argwhere(mol_map == _i).flatten())
                 for _i in set(mol_map)]
        reference = [tuple(np.argwhere(ref_map == _i).flatten())
                     for _i in set(ref_map)]

        for molecule in _list:
            self.assertTrue(
              molecule in reference,
              'mol-ID of atoms {} does not correspond to reference: {}'.format(
                            molecule,
                            ref_map[list(molecule)])
                            )

    def test_assign_molecule(self):
        n_mol = 1
        n_atoms = 9
        molecule = np.zeros((n_atoms))
        neigh_map = [
                [1, 2, 6],
                [0],
                [0],
                [4, 8],
                [3],
                [6, 7],
                [0, 5],
                [5],
                [3],
                ]
        atom_count = n_atoms

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 0, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 1, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 2, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 3, atom_count)
        self.assertEqual(atoms, 6)
        self.assertListEqual(ass.tolist(), [0, 0, 0, 1, 1, 0, 0, 0, 1])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 4, atom_count)
        self.assertEqual(atoms, 6)
        self.assertListEqual(ass.tolist(), [0, 0, 0, 1, 1, 0, 0, 0, 1])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 5, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 6, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 7, atom_count)
        self.assertEqual(atoms, 3)
        self.assertListEqual(ass.tolist(), [1, 1, 1, 0, 0, 1, 1, 1, 0])

        molecule = np.zeros((n_atoms))
        ass, atoms = dissection.assign_molecule(molecule, n_mol, n_atoms,
                                                neigh_map, 8, atom_count)
        self.assertEqual(atoms, 6)
        self.assertListEqual(ass.tolist(), [0, 0, 0, 1, 1, 0, 0, 0, 1])

    def test_read_topology_file(self):
        _d = dissection.read_topology_file(self.dir + '/test_long_mol.xyz')
        self.assertIsInstance(_d, dict)


class TestDistribution(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # rdf


class TestMotion(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_linear_momenta(self):
        _vel = np.ones((2, 3))
        _vel[0] = [1.0, 0.5, 0.25]
        _vel[1] = [-2.0, -1.0, 0.5]
        _l = motion.linear_momenta(_vel, (2, 1))
        self.assertListEqual(_l.tolist(), [0., 0., 1.])

        _l = motion.linear_momenta(np.array([_vel, _vel]), (2, 1))
        self.assertListEqual(_l.flatten().tolist(), [0., 0., 1., 0., 0., 1.])

    def test_angular_momenta(self):
        _vel = np.ones((2, 3))
        _pos = np.ones((2, 3))
        _vel[0] = [1.0, 0.5, 0.25]
        _vel[1] = [-2.0, -1.0, 0.5]
        _pos[0] = [0.0, 0.0, 1.0]
        _a = motion.angular_momenta(_pos, _vel, (2, 1))
        self.assertListEqual(_a.tolist(), [0.5, -0.5, 1.])

        _a = motion.angular_momenta(np.array([_pos, _pos]),
                                    np.array([_vel, _vel]),
                                    (2, 1))
        self.assertListEqual(_a.flatten().tolist(), [0.5, -0.5, 1.] * 2)


class TestGrid(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_map_on_posgrid(self):
        # --- Insufficiently tested
        _grid = np.linspace(-32.0, 32.0, 6400)
        _p = np.array([0.])
        _reg1 = grid.map_on_posgrid(_p, _grid,
                                    mode="gaussian", sigma=1./8, dim=1)
        _reg2 = grid.map_on_posgrid(_p, _grid,
                                    mode="lorentzian", gamma=1./8, dim=1)
        self.assertEqual(100., round(_reg1.sum()))
        self.assertEqual(100., round(_reg2.sum()))
