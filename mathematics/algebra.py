#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np


def angle(v1, v2):
    """Angle between two vectors"""
    al = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al, -1.0, 1.0))

    return a_rad


def signed_angle(v1, v2, n):
    """n is the reference/plane normal for angle direction"""
    al = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al, -1.0, 1.0)) *\
        np.sign(np.dot(np.cross(v1, v2), n))

    return a_rad


def angle_from_points(p1, p2, p3):
    """Angle spanned by p1<--p2, p2-->p3"""
    v1 = p1-p2
    v2 = p3-p2
    al = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al, -1.0, 1.0))

    return a_rad


def dihedral(p1, p2, p3, p4):
    '''Definition along given points p1--p2--p3--p4;
       all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    v1 = p1-p2
    v2 = p3-p2
    v3 = p4-p3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v2)
    dr = np.inner(np.cross(n1, n2), v2)
    dr /= np.linalg.norm(dr)

    dih_rad = np.arccos(np.inner(n1, n2) /
                        (np.linalg.norm(n1, axis=-1) *
                        np.linalg.norm(n2, axis=-1)))
    dih_rad *= dr  # Direction of rotation
    return dih_rad


def plane_normal(p1, p2, p3):
    '''Plane spanned by vectors p1<--p2, p2-->p3;
       all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    v1 = p1-p2
    v2 = p3-p2
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)

    return n


def rotation_matrix(v1, v2):
    '''rotate v1 to match v2'''
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)
    n = np.cross(u1, u2)
    V = np.matrix([[0., -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0.]])
    Id = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # unittest: use np.identity ?
    R = Id + V + V**2*(1.-np.dot(u1, u2))/np.linalg.norm(n)**2

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
