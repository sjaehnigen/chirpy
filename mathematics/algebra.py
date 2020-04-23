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


import numpy as np
from scipy.interpolate import RegularGridInterpolator


def vector(*args):
    '''v = p1 - p2'''
    if len(args) == 1:
        p1, p2 = args[0]
    elif len(args) == 2:
        p1, p2 = args
    else:
        raise TypeError('vector() requires 1 or 2 positional arguments!')

    return p1 - p2


def angle(*args):
    """args: v1, v2; angle between two vectors"""
    if len(args) == 1:
        v1, v2 = args[0]
    elif len(args) == 2:
        v1, v2 = args
    else:
        raise TypeError('angle() requires 1 or 2 positional arguments!')

    al = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al, -1.0, 1.0))

    return a_rad


def signed_angle(*args):
    """args: v1, v2, n; n is the reference/plane normal for angle direction"""
    if len(args) == 1:
        v1, v2, n = args[0]
    elif len(args) == 3:
        v1, v2, n = args
    else:
        raise TypeError('signed_angle() requires 1 or 3 positional arguments!')
    a_rad = angle(v1, v2) * np.sign(np.dot(np.cross(v1, v2), n))

    return a_rad


def angle_from_points(*args):
    """Angle spanned by p1<--p2, p2-->p3"""
    if len(args) == 1:
        p1, p2, p3 = args[0]
    elif len(args) == 3:
        p1, p2, p3 = args
    else:
        raise TypeError('angle_from_points() requires 1 or 3 '
                        'positional arguments!')

    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    a_rad = angle(v1, v2)

    return a_rad


def dihedral(*args):
    '''args: p1, p2, p3, p4; dihedral angle along p1<--p2-->p3-->p4;
       all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    if len(args) == 1:
        p1, p2, p3, p4 = args[0]
    elif len(args) == 4:
        p1, p2, p3, p4 = args
    else:
        raise TypeError('dihedral() requires 1 or 4 positional arguments!')

    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    v3 = np.array(p4) - np.array(p3)
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v2)
    dr = np.inner(np.cross(n1, n2), v2)
    dr /= np.linalg.norm(dr)

    dih_rad = np.arccos(np.inner(n1, n2) /
                        (np.linalg.norm(n1, axis=-1) *
                        np.linalg.norm(n2, axis=-1)))
    dih_rad *= dr  # Direction of rotation
    return dih_rad


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
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)

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

    return np.inner(np.cross(v1, v2), v3)


def rotation_matrix(*args):
    '''rotate v1 to match v2'''
    if len(args) == 1:
        v1, v2 = args[0]
    elif len(args) == 2:
        v1, v2 = args
    else:
        raise TypeError('rotation_matrix() requires 1 or 2 '
                        'positional arguments!')

    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    n = np.cross(u1, u2)
    V = np.matrix([[0., -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0.]])
    Id = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # unittest: use np.identity ?
    R = Id + V + V**2 * (1. - np.dot(u1, u2)) / np.linalg.norm(n)**2

    return np.asarray(R)


def change_euclidean_basis(v, basis):
    '''Transform coordinates to cell vector basis with the help of dual basis
       v ... set of vectors of shape (....., 3) in old basis
       basis ... new basis tensor of shape (3, 3)'''
    M = np.zeros_like(basis)
    M[0] = np.cross(basis[1], basis[2])
    M[1] = np.cross(basis[2], basis[0])
    M[2] = np.cross(basis[0], basis[1])
    V = np.dot(basis[0], np.cross(basis[1], basis[2]))

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
