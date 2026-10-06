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

import os
import unittest
import numpy as np
from functools import partial
from itertools import product

from chirpy.topology import mapping, dissection, motion, grid  # , distribution
from chirpy.topology import distribution
from chirpy.read import coordinates
from chirpy import constants

_test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_files'


class TestMapping(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dist_crit_aa(self):
        symbols = tuple([_s[0] for _s in constants._rvdw_list])[:-1]
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
        dec = mapping.dec(a, i, n_ind=4)

        self.assertIsInstance(dec, list)
        self.assertEqual(len(dec), max(i) + 1)
        self.assertTrue(np.allclose(
                       np.concatenate(
                         tuple([_d.flatten() for _d in dec])).reshape((20, 3)),
                       a
                       ))

        # --- repeat without interpreting numerically
        dec = mapping.dec(a, i)
        self.assertEqual(len(dec), 2)
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

    def test_cowt_with_mask(self):
        # --- 4 points, grouped into 2 units of 2 points each
        pos = np.array([
                [1., 1., 0.],
                [2., 1., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                ])
        wt = np.ones(4)
        mask = [0, 0, 1, 1]
        cowt = mapping.cowt(pos, wt, mask=mask)
        self.assertTupleEqual(cowt.shape, (2, 3))
        self.assertListEqual(
                np.around(cowt, decimals=6).tolist(),
                [[1.5, 1., 0.], [0.5, 0., 0.]]
                )

    def test_cell_l_deg(self):
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        cell_vec_aa = mapping.cell_vec(cell_aa_deg)
        _rt = mapping.cell_l_deg(cell_vec_aa)
        self.assertListEqual(
                np.around(_rt, decimals=3).tolist(),
                np.around(cell_aa_deg, decimals=3).tolist()
                )

        # --- multiply scales lengths but not angles
        _rt2 = mapping.cell_l_deg(cell_vec_aa, multiply=(2, 1, 1))
        self.assertAlmostEqual(_rt2[0], 2 * cell_aa_deg[0], places=3)
        self.assertListEqual(
                np.around(_rt2[3:], decimals=3).tolist(),
                np.around(cell_aa_deg[3:], decimals=3).tolist()
                )

    def test_cell_vec(self):
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        cell_vec_aa = np.around(mapping.cell_vec(cell_aa_deg), decimals=3)
        self.assertListEqual(
                cell_vec_aa.tolist(),
                [[24.218, 0., 0.], [0., 15.92, 0.], [-4.995, 0., 12.393]]
                )
        cell_vec_aa = np.around(mapping.cell_vec(cell_aa_deg,
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

    def test_get_cell_coordinates(self):
        # --- l/rtransform: convert between Cartesian and cell-vector
        #     (fractional) basis for a non-tetragonal (monoclinic) cell
        cell_aa_deg = np.array([24.218, 15.92, 13.362, 90.0, 111.95, 90.0])
        pos_cart = np.array([[5., 4., 3.], [1., 2., 3.]])

        pos_cell = mapping.get_cell_coordinates(pos_cart, cell_aa_deg)
        pos_back = mapping.get_cartesian_coordinates(pos_cell, cell_aa_deg)
        self.assertTrue(np.allclose(pos_back, pos_cart))

        # --- angular quantities (e.g. moments/tensors) transform via the
        # reciprocal (dual) basis instead of the direct one
        vec_cart = np.array([1., 0., 0.])
        vec_cell = mapping.get_cell_coordinates(vec_cart, cell_aa_deg,
                                                angular=True)
        vec_back = mapping.get_cartesian_coordinates(vec_cell, cell_aa_deg,
                                                      angular=True)
        self.assertTrue(np.allclose(vec_back, vec_cart))

    def test_wrap_pbc(self):
        cell_aa_deg = np.array([1., 2., np.sqrt(2), 90., 135., 90.])
        _p1 = np.ones((2, 3))
        _p1[0] = [0.99, 1., 0.5]
        _p1[1] = [1.1, 2.1, 1.1]
        _p1 = np.around(mapping.wrap_pbc(_p1, cell_aa_deg), decimals=3)
        self.assertListEqual(_p1.flatten().tolist(),
                             [-0.01, 1., 0.5, 0.1, 0.1, 0.1])
        _shape = (3, 2, 3)
        _p2 = np.ones(_shape)
        self.assertTupleEqual(mapping.wrap_pbc(_p2, cell_aa_deg).shape, _shape)

    def test_distance_pbc(self):
        cell_aa_deg = np.array([1., 2., np.sqrt(2), 90., 135., 90.])
        _d = np.around(mapping.distance_pbc(np.array([0.99, 1., 0.5]),
                                            np.array([1.1, 2.1, 1.1]),
                                            cell=cell_aa_deg),
                       decimals=3)
        self.assertListEqual(_d.tolist(), [0.11, -0.9, -0.4])

        _d = np.around(mapping.distance_pbc(np.array([0.15, 1.0, 0.5]),
                                            np.array([0.85, 1.0, 0.5]),
                                            cell=cell_aa_deg),
                       decimals=3)
        self.assertListEqual(_d.tolist(), [-0.3, 0., 0.])

        _d = np.around(mapping.distance_pbc(np.array([0.0, 0.0, 0.0]),
                                            np.array([0.26, 0.0, 0.49]),
                                            cell=cell_aa_deg,
                                            mode='priority'),
                       decimals=3)
        self.assertListEqual(_d.tolist(), [0.26, 0.0, 0.49])

        # specific test for monoclinic and triclinic cells
        # requires positive test for cell_vec

        for cell_aa_deg in [
                np.array([1., 2., np.sqrt(2), 90., 135., 90.]),
                np.array([1., 1., 1., 90., 90., 120.]),
                np.array([10., 2., np.sqrt(2), 30., 135., 90.]),
                np.array([3., 8., np.sqrt(2), 30., 135., 20.]),
                ]:
            cell_vec = mapping.cell_vec(cell_aa_deg)
            n_samples = 10
            samples = np.random.uniform(0, 1, size=(n_samples, 3))
            D_test = mapping.distance_matrix(samples, cell=cell_aa_deg,
                                             mode='accurate')
            D_ref = np.amin([mapping.distance_matrix(
                                    samples + _L @ cell_vec,
                                    samples,
                                    cell=None,
                            ) for _L in product([0, 1, -1], repeat=3)], axis=0)
            self.assertListEqual(
                    np.round(D_test, decimals=3).flatten().tolist(),
                    np.round(D_ref, decimals=3).flatten().tolist(),
                    f'triclinic MIC failed for cell {cell_aa_deg}'
                    )

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

    def test_angle_pbc(self):
        cell_aa_deg = np.array([10., 10., 10., 90., 90., 90.])
        p0 = np.array([1., 0., 0.])
        p1 = np.array([0., 0., 0.])
        p2 = np.array([0., 1., 0.])

        # --- unsigned angle: 90 degrees
        _a = mapping.angle_pbc(p0, p1, p2, cell=cell_aa_deg)
        self.assertAlmostEqual(np.degrees(_a), 90.0, places=6)

        # --- unsigned angle is invariant under swapping the outer points
        _a_swap = mapping.angle_pbc(p2, p1, p0, cell=cell_aa_deg)
        self.assertAlmostEqual(_a, _a_swap, places=6)

        # --- signed angle with explicit plane normal distinguishes
        # orientation (does not crash, cf. former TypeError bug)
        _a_signed = mapping.angle_pbc(p0, p1, p2, cell=cell_aa_deg,
                                      signed=True,
                                      plane_normal=np.array([0., 0., 1.]))
        _a_signed_rev = mapping.angle_pbc(p2, p1, p0, cell=cell_aa_deg,
                                          signed=True,
                                          plane_normal=np.array([0., 0., 1.]))
        self.assertAlmostEqual(_a_signed, -_a_signed_rev, places=6)

        # --- signed angle without an explicit plane normal still works
        # (falls back to cross(v0, v1), degenerating to the unsigned value)
        _a_signed_default = mapping.angle_pbc(p0, p1, p2, cell=cell_aa_deg,
                                              signed=True)
        self.assertAlmostEqual(abs(_a_signed_default), _a, places=6)

    def test_unwrap_pbc(self):
        cell_aa_deg = np.array([10., 10., 10., 90., 90., 90.])
        # --- 2 frames of a single atom crossing the x boundary
        pos = np.array([
                [[9.5, 5., 5.]],
                [[0.5, 5., 5.]],
                ])
        _unwrapped = mapping.unwrap_pbc(pos, cell=cell_aa_deg)
        self.assertAlmostEqual(_unwrapped[1, 0, 0], 10.5, places=6)
        self.assertListEqual(_unwrapped[0].tolist(), pos[0].tolist())

    def test_mean_pbc(self):
        cell_aa_deg = np.array([10., 10., 10., 90., 90., 90.])
        # --- two points straddling the periodic boundary along x
        pos = np.array([[9.5, 5., 5.], [0.5, 5., 5.]])
        _mean = mapping.mean_pbc(pos, cell=cell_aa_deg, wrap=True)
        self.assertAlmostEqual(_mean[0] % 10., 0.0, places=6)
        self.assertAlmostEqual(_mean[1], 5.0, places=6)

        # --- points well within the box: mean_pbc matches the naive mean
        pos2 = np.array([[4., 5., 5.], [6., 5., 5.]])
        _mean2 = mapping.mean_pbc(pos2, cell=cell_aa_deg, wrap=True)
        self.assertListEqual(np.around(_mean2, decimals=6).tolist(),
                             [5., 5., 5.])

    def test_neighbour_matrix(self):
        # --- water molecule: O bonded to both H, H-H not bonded
        pos = np.array([
                [0.000, 0.000, 0.000],
                [0.958, 0.000, 0.000],
                [-0.240, 0.927, 0.000],
                ])
        symbols = ('O', 'H', 'H')
        _neigh = mapping.neighbour_matrix(pos, symbols)
        self.assertTupleEqual(_neigh.shape, (3, 3))
        self.assertTrue(_neigh[0, 1])
        self.assertTrue(_neigh[0, 2])
        self.assertTrue(_neigh[1, 0])
        self.assertTrue(_neigh[2, 0])
        self.assertFalse(_neigh[1, 2])
        self.assertFalse(_neigh[2, 1])

    def test_connectivity(self):
        pos = np.array([
                [0.000, 0.000, 0.000],
                [0.958, 0.000, 0.000],
                [-0.240, 0.927, 0.000],
                ])
        symbols = ('O', 'H', 'H')
        _con = mapping.connectivity(pos, symbols)
        self.assertEqual(len(_con), 3)
        self.assertListEqual(sorted(_con[0].tolist()), [1, 2])
        self.assertListEqual(_con[1].tolist(), [0])
        self.assertListEqual(_con[2].tolist(), [0])

    def test_nearest_neighbour(self):
        p0 = np.array([[0., 0., 0.], [5., 0., 0.]])
        p1 = np.array([[0.1, 0., 0.], [4.9, 0., 0.], [10., 10., 10.]])
        _nn = mapping.nearest_neighbour(p0, p1)
        self.assertListEqual(_nn.tolist(), [0, 1])

        _nn_ignore = mapping.nearest_neighbour(p0, p1, ignore=[0])
        self.assertListEqual(_nn_ignore.tolist(), [1, 1])

        _nn_dist, _dist = mapping.nearest_neighbour(p0, p1,
                                                    return_distances=True)
        self.assertListEqual(_nn_dist.tolist(), [0, 1])
        self.assertListEqual(np.around(_dist, decimals=6).tolist(),
                             [0.1, 0.1])

    def test_close_neighbours(self):
        # --- two atoms coincide (duplicate), one is far away
        p0 = np.array([
                [0., 0., 0.],
                [0.00001, 0., 0.],
                [5., 5., 5.],
                ])
        _close = mapping.close_neighbours(p0, crit=1.E-3)
        self.assertEqual(len(_close), 1)
        self.assertEqual(_close[0][0], 0)
        self.assertEqual(_close[0][1][0][0], 1)

        # --- default crit from symbols is much stricter than a bond length
        symbols = ('O', 'O', 'O')
        _close_bonded = mapping.close_neighbours(
                np.array([[0., 0., 0.], [0.96, 0., 0.], [5., 5., 5.]]),
                symbols=symbols
                )
        self.assertListEqual(_close_bonded, [])

    def test_join_molecules(self):
        cell_aa_deg = np.array([10., 10., 10., 90., 90., 90.])
        # --- water molecule with one H wrapped across the x boundary
        # (unwrapped bond length O-H1 is exactly 0.958 angstrom)
        pos = np.array([
                [0.200, 5.000, 5.000],
                [9.242, 5.000, 5.000],
                [0.446, 5.927, 5.000],
                ])
        symbols = ('O', 'H', 'H')
        mol_map = [0, 0, 0]

        joined, mol_com_aa = mapping.join_molecules(
                pos, mol_map, cell_aa_deg, symbols=symbols)

        # --- after joining, O-H1 distance should be reconstructed to the
        # genuine (un-wrapped) bond length
        _d = np.linalg.norm(joined[0] - joined[1])
        self.assertAlmostEqual(_d, 0.958, places=2)
        self.assertTupleEqual(mol_com_aa.shape, (1, 3))

    def test_find_methyl_groups(self):
        # --- a methyl-like CH3 group plus an unrelated H
        pos = np.array([
                [0.000, 0.000, 0.000],
                [1.090, 0.000, 0.000],
                [-0.363, 1.028, 0.000],
                [-0.363, -0.514, 0.890],
                [10.0, 10.0, 10.0],
                ])
        symbols = ('C', 'H', 'H', 'H', 'H')
        import io
        import contextlib
        _out = io.StringIO()
        with contextlib.redirect_stdout(_out):
            mapping.find_methyl_groups(pos, symbols)
        self.assertIn('A001 A002 A003 A004', _out.getvalue())

    def test_ishydrogenbond(self):
        cell_aa_deg = np.array([20., 20., 20., 90., 90., 90.])
        # --- linear O-H...O hydrogen bond
        positions = np.array([
                [0.000, 0.000, 0.000],   # 0: donor O
                [0.960, 0.000, 0.000],   # 1: H
                [2.860, 0.000, 0.000],   # 2: acceptor O
                [10.0, 10.0, 10.0],      # 3: far, unrelated O
                ])
        _hb = mapping.ishydrogenbond(positions, [0], [2, 3], [1],
                                     cell=cell_aa_deg)
        self.assertTupleEqual(_hb.shape, (1, 2))
        self.assertTrue(_hb[0, 0])
        self.assertFalse(_hb[0, 1])

        # --- bent geometry (off-axis) breaks the angle criterion
        positions_bent = positions.copy()
        positions_bent[2] = [2.0, 2.0, 0.0]
        _hb_bent = mapping.ishydrogenbond(positions_bent, [0], [2], [1],
                                          cell=cell_aa_deg)
        self.assertFalse(_hb_bent[0, 0])

    def test_guess_atom_types(self):
        # --- two water molecules: atoms of the same chemical environment
        # get the same (arbitrary) integer type
        pos = np.array([
                [0.000, 0.000, 0.000],
                [0.958, 0.000, 0.000],
                [-0.240, 0.927, 0.000],
                [5.000, 0.000, 0.000],
                [5.958, 0.000, 0.000],
                [4.760, 0.927, 0.000],
                ])
        symbols = ('O', 'H', 'H', 'O', 'H', 'H')

        for order in (0, 1, 2):
            _types = mapping.guess_atom_types(pos, symbols, order=order)
            self.assertEqual(len(_types), 6)
            # --- both oxygens share a type, all four hydrogens share a type
            self.assertEqual(_types[0], _types[3])
            self.assertEqual(_types[1], _types[2])
            self.assertEqual(_types[1], _types[4])
            self.assertEqual(_types[1], _types[5])
            self.assertNotEqual(_types[0], _types[1])


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
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 8, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 2, 2, 3, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 6, 6, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 9, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2,
            2, 2, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 13, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            9, 9, 9, 9, 9, 8, 8, 8, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 15, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3,
            3, 3, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 5, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 7, 7,
            7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 9, 9, 9, 10, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 11,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            0, 0, 2, 2, 12, 4, 13, 5, 6, 6, 8, 8, 14, 10, 15, 11, 1, 1, 3, 3,
            4, 12, 5, 13, 7, 7, 9, 9, 10, 14, 11, 15
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
                            ref_map[list(molecule)]),
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

    def test_radial_distribution_function(self):
        # --- a spatially uniform ("ideal gas") distribution of particles
        # has a flat RDF equal to 1 (no structure)
        np.random.seed(42)
        n_frames = 5
        n_particles = 2000
        length = 20.0
        cell = np.array([length, length, length, 90., 90., 90.])

        positions = np.random.uniform(0, length,
                                      size=(n_frames, n_particles, 3))
        origins = np.random.uniform(0, length, size=(n_frames, 1, 3))

        r, rdf = distribution.radial_distribution_function(
                positions, origins, cell=cell, rng=(0.5, 8.0), bins=40)

        self.assertTupleEqual(r.shape, (40,))
        self.assertTupleEqual(rdf.shape, (40,))
        # --- averaged over many bins/frames, RDF should fluctuate around 1
        self.assertAlmostEqual(rdf.mean(), 1.0, delta=0.1)


class TestMotion(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/topology'

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

    def test_hydrogen_bond_lifetime_analysis(self):
        positions = coordinates.xyzReader(self.dir + '/positions.xyz')[0]
        hb_acf_con = np.round(motion.hydrogen_bond_lifetime_analysis(
                                positions,
                                *([2, 3], np.arange(4, 126), [0, 1]),
                                dist_crit=2.8,
                                angle_crit=130,
                                cell=np.array([20., 14., 14., 90., 90., 90.]),
                                mode='continuous',
                                no_average=False
                                ), decimals=3)
        ref = np.loadtxt(self.dir + '/hb_lifetime.dat')

        self.assertListEqual(hb_acf_con.tolist(), ref.tolist())


class TestGrid(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_regularisation(self):
        # --- Insufficiently tested
        _grid = np.linspace(-32.0, 32.0, 6400)
        _p = np.array([0.])
        _reg1 = grid.regularisation(_p, _grid, 1./8, mode="gaussian")
        _reg2 = grid.regularisation(_p, _grid, 1./8, mode="lorentzian")
        self.assertEqual(100., round(_reg1.sum()))
        self.assertEqual(100., round(_reg2.sum()))

        _w = 0.1
        X = np.linspace(0, 1, 101)
        P = np.array([[0.5]])
        _reg1 = grid.regularisation(P, X, _w, mode="lorentzian_std")[0]
        _reg2 = grid.regularisation(P, X, _w, mode="gaussian_std")[0]
        self.assertAlmostEqual(float(_reg1[X == 0.5]), 1.0)
        self.assertAlmostEqual(float(_reg1[X == _w/2 + 0.5]), 0.5)
        self.assertAlmostEqual(float(_reg2[X == 0.5]), 1.0)
        self.assertAlmostEqual(float(_reg2[X == _w/2 + 0.5]), 0.5)

        # --- 3D Gaussian
        X = np.linspace(0, 1, 100)
        _grid = np.array(np.meshgrid(X, X, X, indexing='ij'))
        _P = np.array([
            [0.5, 0.5, 0.5],
        ])

        _reg1 = grid.regularisation(_P, _grid, 0.05, mode="gaussian")

        self.assertAlmostEqual(0.97, _reg1.sum()/100**3, delta=0.03)
