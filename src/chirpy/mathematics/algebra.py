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


import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..constants import eijk


def dot(vector0, vector1):
    '''v0 · v1 with vectors v0/v1 of shape ([n_frames, n_units], 3)
       (numpy.newaxis accepted)

       returns vector with the same shape as v0/v1
       '''
    v0 = np.array(vector0)
    v1 = np.array(vector1)

    if (s0 := len(v0.shape)) != (s1 := len(v1.shape)):
        raise ValueError('operands cannot be broadcast together with shapes '
                         f'{s0} {s1}')

    if s0 == 1:
        return np.dot(v0, v1)
    else:
        return np.sum(v0*v1, axis=-1)


def cross(vector0, vector1):
    '''v0 × v1 with vectors v0/v1 of shape ([n_frames, n_units], 3)
       (numpy.newaxis accepted)

       returns vector with the same shape as v0/v1
       '''
    v0 = np.array(vector0)
    v1 = np.array(vector1)

    if (s0 := len(v0.shape)) != (s1 := len(v1.shape)):
        raise ValueError('operands cannot be broadcast together with shapes '
                         f'{s0} {s1}')

    if s0 == 1:
        return np.cross(v0, v1)
    else:
        return np.sum(eijk
                      * v0[..., :, None, None]
                      * v1[..., None, :, None],
                      axis=(s0-1, s0))


def vector(*args):
    '''v = p1 - p0'''
    if len(args) == 1:
        p0, p1 = args[0]
    elif len(args) == 2:
        p0, p1 = args
    else:
        raise TypeError('vector() requires 1 or 2 positional arguments!')

    return p1 - p0


def angle(*args):
    """args: v0, v1; angle between two vectors"""
    if len(args) == 1:
        v0, v1 = args[0]
    elif len(args) == 2:
        v0, v1 = args
    else:
        raise TypeError('angle() requires 1 or 2 positional arguments!')

    norm0 = np.linalg.norm(v0, axis=-1)
    norm1 = np.linalg.norm(v1, axis=-1)
    al = dot(v0, v1)/norm0/norm1
    a_rad = np.arccos(np.clip(al, -1.0, 1.0))

    return a_rad


def signed_angle(*args):
    """args: v0, v1, n; n is the reference/plane normal for angle direction;
       sign: v0 --(rot)-> v1"""
    if len(args) == 1:
        v0, v1, n = args[0]
    elif len(args) == 3:
        v0, v1, n = args
    else:
        raise TypeError('signed_angle() requires 1 or 3 positional arguments!')
    a_rad = angle(v0, v1) * np.sign(dot(cross(v0, v1), n))

    return a_rad


def angle_from_points(*args):
    """Angle spanned by p1<--p2, p2-->p3"""
    if len(args) == 1:
        p0, p1, p2 = args[0]
    elif len(args) == 3:
        p0, p1, p2 = args
    else:
        raise TypeError('angle_from_points() requires 1 or 3 '
                        'positional arguments!')

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    a_rad = angle(v0, v1)

    return a_rad


def dihedral(*args):
    '''args: v0, v1, v2; dihedral angle along <-v0-.-v1->.-v2->;
       all v as 3d-np.vectors or np.arrays (last axis will be used)'''
    if len(args) == 1:
        v0, v1, v2 = args[0]
    elif len(args) == 3:
        v0, v1, v2 = args
    else:
        raise TypeError('dihedral() requires 1 or 3 positional arguments!')

    n0 = cross(v0, v1)
    n1 = cross(v2, v1)
    dr = np.sign(dot(cross(n0, n1), v1))

    norm0 = np.linalg.norm(n0, axis=-1)
    norm1 = np.linalg.norm(n1, axis=-1)

    dih_rad = np.arccos(dot(n0, n1)/norm0/norm1)
    dih_rad *= dr  # Direction of rotation

    return dih_rad


def dihedral_from_points(*args):
    '''args: p0, p1, p2, p3; dihedral angle along p0<--p1-->p2-->p3;
       all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    if len(args) == 1:
        p0, p1, p2, p3 = args[0]
    elif len(args) == 4:
        p0, p1, p2, p3 = args
    else:
        raise TypeError('dihedral_from_points() requires 1 or 4 positional '
                        'arguments!')

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)

    return dihedral(v0, v1, v2)


def plane_normal(*args):
    '''Plane spanned by vectors p1<--p2, p2-->p3;
       all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    if len(args) == 1:
        p1, p2, p3 = args[0]
    elif len(args) == 3:
        p1, p2, p3 = args
    else:
        raise TypeError('plane_normal() requires 1 or 3 positional arguments!')

    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n = cross(v1, v2)
    n /= np.linalg.norm(n, axis=-1)[..., None]

    return n


def triple_product(*args):
    '''t = (v1 x v2) · v3'''
    if len(args) == 1:
        v1, v2, v3 = args[0]
    elif len(args) == 3:
        v1, v2, v3 = args
    else:
        raise TypeError('triple_product() requires 1 or 3 '
                        'positional arguments!')

    return dot(cross(v1, v2), v3)


def rotation_matrix(*args, angle=None):
    '''rotate v1 to match v2 or normal vector (requires angle)'''
    if len(args) == 1:
        n = args[0]
        nnorm = np.linalg.norm(n)
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle) / nnorm

    elif len(args) == 2:
        v1, v2 = args
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        n = cross(u1, u2)
        nnorm = np.linalg.norm(n)
        cos_ang = dot(u1, u2)
        sin_ang = 1.0
    else:
        raise TypeError('rotation_matrix() requires 1 or 2 '
                        'positional arguments!')

    V = np.matrix([
          [0., -n[2], n[1]],
          [n[2], 0, -n[0]],
          [-n[1], n[0], 0.]
        ])

    U = np.matrix([
          [n[0]**2, n[0]*n[1], n[0]*n[2]],
          [n[0]*n[1], n[1]**2, n[1]*n[2]],
          [n[0]*n[2], n[1]*n[2], n[2]**2],
        ]) / nnorm**2

    Id = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    # unittest: use np.identity ?
    R = Id*cos_ang + V*sin_ang + U*(1.-cos_ang)

    return np.asarray(R)


def change_euclidean_basis(v, basis):
    '''Transform coordinates to cell vector basis with the help of dual basis
       v ... set of vectors of shape (....., 3) in old basis
       basis ... new basis tensor of shape (3, 3)'''
    M = np.zeros_like(basis)
    M[0] = cross(basis[1], basis[2])
    M[1] = cross(basis[2], basis[0])
    M[2] = cross(basis[0], basis[1])
    V = dot(basis[0], cross(basis[1], basis[2]))

    # direction cosine
    return 1 / V * np.tensordot(v, M, axes=(-1, 1))


def kabsch_algorithm(P, ref):
    '''Align P with respect to ref. Returns a rotation matrix'''
    C = np.dot(np.transpose(ref), P)
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    # Create Rotation matrix U
    return np.dot(V, W)


def rotate_vector(vector, R, origin=np.zeros(3)):
    '''Rotate vector around given origin with rotation matrix R.
       Returns new vector.

       R of shape (N, N)
       vector of shape (N) or (M, N).
    '''

    if len(vector.shape) == 1:
        return np.einsum('ji, i -> j', R, vector - origin) + origin
    elif len(vector.shape) == 2:
        return np.einsum('ji, mi -> mj', R, vector - origin) + origin


def rotate_griddata(grid_positions, grid_data, R, origin=np.zeros(3)):
    '''Rotate scalar field around given origin with rotation matrix R.
       Keeps grid_positions fixed. Returns new grid_data.

       R of shape (N, N)
       grid_positions with shape (N, X, Y, Z)
       grid_data of shape (X, Y, Z)
       '''

    _p_grid = grid_positions - origin[:, None, None, None]
    _f = RegularGridInterpolator(
              (_p_grid[0, :, 0, 0],
               _p_grid[1, 0, :, 0],
               _p_grid[2, 0, 0, :]),
              grid_data,
              bounds_error=False,
              fill_value=0.0
              )
    # --- unclear why it has to be the other way round (ij)
    _new_p_grid = np.einsum('ij, imno -> mnoj', R, _p_grid)

    return _f(_new_p_grid)
