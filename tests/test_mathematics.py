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
import numpy as np

from chirpy.mathematics import algebra


class TestAlgebra(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_angle(self):
        ang = algebra.angle([0.51, 0.51, 0.], [1., 0., 0.])
        self.assertEqual(ang, 45 * np.pi / 180.)

    def test_signed_angle(self):
        ang = algebra.signed_angle([0.51, 0.51, 0.], [1, 0, 0], [0, 0, 1])
        self.assertEqual(ang, -45 * np.pi / 180.)

    def test_cross(self):
        # --- single vector pair matches numpy
        v0 = np.array([1., 0., 0.])
        v1 = np.array([0., 1., 0.])
        self.assertListEqual(algebra.cross(v0, v1).tolist(),
                             np.cross(v0, v1).tolist())

        # --- batched vectors (shape (n, 3)) also match numpy row-wise
        v0b = np.array([[1., 0., 0.], [0., 1., 0.]])
        v1b = np.array([[0., 1., 0.], [1., 0., 0.]])
        self.assertListEqual(algebra.cross(v0b, v1b).tolist(),
                             np.cross(v0b, v1b).tolist())

        # --- shape mismatch raises
        with self.assertRaises(ValueError):
            algebra.cross(v0, v1b)

    def test_angle_from_points(self):
        ang = algebra.angle_from_points([0, 0, 0], [2, 0, 1], [2, 2, 1])
        self.assertEqual(ang, 90 * np.pi / 180)

    def test_dihedral(self):
        # --- vectors derived from the same geometry as
        #     test_dihedral_from_points, dihedral() should give the same
        #     angle since dihedral_from_points() internally calls dihedral()
        p0, p1, p2, p3 = [0, 0, 0], [0, 0.5, 0], [0, 0.5, 0.5], \
            [-0.2, 0.3, 0.5]
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        dih = algebra.dihedral(v0, v1, v2)
        self.assertEqual(np.round(dih * 180 / np.pi, decimals=6), -45)

        # --- single tuple-of-3 argument form gives an identical result
        dih_tuple = algebra.dihedral((v0, v1, v2))
        self.assertEqual(dih, dih_tuple)

        # --- sign flips when the rotational sense is reversed
        dih_flipped = algebra.dihedral(-v0, v1, v2)
        self.assertEqual(np.round(dih_flipped * 180 / np.pi, decimals=6),
                         135)

        # --- wrong number of arguments raises
        with self.assertRaises(TypeError):
            algebra.dihedral(v0, v1)

    def test_dihedral_from_points(self):
        dih = algebra.dihedral_from_points(
                [0, 0, 0],
                [0, 0.5, 0],
                [0, 0.5, 0.5],
                [-0.2, 0.3, 0.5],
                )
        self.assertEqual(np.round(dih * 180 / np.pi, decimals=6), -45)

    def test_plane_normal(self):
        _n = algebra.plane_normal([0, 0, 0.], [2, 0, 1], [2, 2, 1])
        self.assertListEqual(np.around(_n, decimals=7).tolist(),
                             [0.4472136, 0., -0.8944272])

    def test_rotation_matrix(self):
        for _v1, _v2 in [
           (np.array([0.123, -123.923, 7.219]), np.array([-13., 0., 0.1289])),
           (np.array([0.123, -123.923, 7.219]), np.array([-13., 11., 0.1289])),
           (np.array([0.123, -12.923, 7.219]), np.array([-13., 1., -0.1289])),
           ]:
            R = algebra.rotation_matrix(_v1, _v2)
            self.assertAlmostEqual(
                             algebra.angle(R @ _v1, _v2),
                             0.0,
                             7,
                             f'rotation matrix fails for {_v1}, {_v2}'
                             )
            self.assertEqual(*map(
                            lambda x: np.around(np.linalg.norm(x), decimals=9),
                            [R @ _v1, _v1]
                            ))

    def test_rotation_matrix_axis_angle(self):
        # --- rotation by 90 degrees about the z-axis maps x --> y
        R = algebra.rotation_matrix([0., 0., 1.], angle=np.pi / 2)
        self.assertListEqual(
                np.around(R @ np.array([1., 0., 0.]), decimals=9).tolist(),
                [0., 1., 0.]
                )

        # --- zero angle is the identity, independent of the axis
        for _n in [[1., 0., 0.], [0.123, -1.2, 7.5]]:
            R0 = algebra.rotation_matrix(_n, angle=0.0)
            self.assertListEqual(np.around(R0, decimals=9).tolist(),
                                 np.identity(3).tolist())

        # --- a full turn (2*pi) about any axis is also the identity
        R_full = algebra.rotation_matrix([0.3, -1.1, 2.4], angle=2 * np.pi)
        self.assertListEqual(np.around(R_full, decimals=7).tolist(),
                             np.identity(3).tolist())

        # --- rotation preserves the vector norm and the axis itself
        _n = np.array([0.4, -0.9, 3.2])
        _v = np.array([1.5, -2.3, 0.7])
        R = algebra.rotation_matrix(_n, angle=1.234)
        self.assertAlmostEqual(np.linalg.norm(R @ _v), np.linalg.norm(_v), 9)
        self.assertListEqual(np.around(R @ _n, decimals=7).tolist(),
                             np.around(_n, decimals=7).tolist())

        # --- consistent with the two-vector form: rotating v1 onto v2 by
        # the axis-angle form (axis = v1 x v2, angle = angle(v1, v2)) must
        # give the same result as the two-vector form
        _v1 = np.array([0.123, -123.923, 7.219])
        _v2 = np.array([-13., 11., 0.1289])
        _axis = algebra.cross(_v1, _v2)
        _angle = algebra.angle(_v1, _v2)
        R_2v = algebra.rotation_matrix(_v1, _v2)
        R_aa = algebra.rotation_matrix(_axis, angle=_angle)
        self.assertListEqual(np.around(R_2v, decimals=7).tolist(),
                             np.around(R_aa, decimals=7).tolist())

    def test_change_euclidean_basis(self):
        # --- insufficiently tested (because method will be extended to modes)
        eb = algebra.change_euclidean_basis(
                   [[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]],
                   np.identity(3)[::-1],
                   )
        self.assertListEqual(eb.flatten().tolist(),
                             [0., 0., 1., 0., 1., 0., 1., 0., 0.])

    def test_kabsch_algorithm(self):
        _p = np.array([
                [0, 0, 0],
                [0, 0.5, 0],
                [0, 0.5, 0.5],
                [-0.2, 0.3, 0.5],
                ])
        _v1 = np.array([0.123143, -123.923, 7.219])
        _v2 = np.array([-13., 0., 0.128923518])
        R = algebra.rotation_matrix(_v1, _v2)

        _p2 = np.tensordot(R, _p, axes=([1, -1])).T
        R2 = algebra.kabsch_algorithm(_p2, _p)

        self.assertTrue(np.allclose(np.matmul(R, R2), np.identity(3)),
                        'The rotation matrices are not inverse!')

    def test_rotate_griddata(self):
        n = 21
        coords = np.linspace(-5, 5, n)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        grid_positions = np.array([X, Y, Z])
        data = np.zeros((n, n, n))
        ix = np.argmin(np.abs(coords - 2.0))
        iy = np.argmin(np.abs(coords - 0.0))
        iz = np.argmin(np.abs(coords - 0.0))
        data[ix, iy, iz] = 1.0

        # --- rotate the peak by 90 degrees around the z axis: (2,0,0) ->
        # (0, 2, 0)
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta), 0.],
                     [np.sin(theta), np.cos(theta), 0.],
                     [0., 0., 1.]])
        rotated = algebra.rotate_griddata(grid_positions, data, R)

        peak = np.unravel_index(np.argmax(rotated), rotated.shape)
        self.assertAlmostEqual(coords[peak[0]], 0.0, places=6)
        self.assertAlmostEqual(coords[peak[1]], 2.0, places=6)
        self.assertAlmostEqual(coords[peak[2]], 0.0, places=6)
        self.assertAlmostEqual(rotated[peak], 1.0, places=6)


class TestAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_divrot(self):
        from chirpy.mathematics.analysis import divrot

        n = 20
        length = 10.0
        cell_vec = np.eye(3) * (length / n)
        coords = (np.arange(n) - n / 2) * (length / n)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # --- radial field F(r) = r: analytically div(F) = 3, rot(F) = 0
        data = np.array([X, Y, Z])
        div, rot = divrot(data, cell_vec)

        self.assertTupleEqual(div.shape, (n, n, n))
        self.assertTupleEqual(rot.shape, (3, n, n, n))

        # --- exclude the outer boundary where the finite-difference
        # gradient is inaccurate
        _c = slice(5, 15)
        self.assertAlmostEqual(div[_c, _c, _c].mean(), 3.0, places=6)
        self.assertAlmostEqual(
                np.abs(rot[:, _c, _c, _c]).max(), 0.0, places=6)
