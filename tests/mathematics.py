# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
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

    def test_angle_from_points(self):
        ang = algebra.angle_from_points([0, 0, 0], [2, 0, 1], [2, 2, 1])
        self.assertEqual(ang, 90 * np.pi / 180)

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
