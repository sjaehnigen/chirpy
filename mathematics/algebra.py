#!/usr/bin/env python3

import numpy as np

def rotation_matrix(v1, v2):
    '''rotate v1 to match v2'''
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)  
    n = np.cross(u1,u2)
    V = np.matrix([[0, -n[2], n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    R = I + V +V**2*(1-np.dot(u1,u2))/np.linalg.norm(n)**2

    return np.asarray( R ) #this matrix/array dualism may lead to trouble

def change_euclidean_basis(v, basis):
    '''Transform coordinates to cell vector basis with the help of dual basis
    v ... set of vectors of shape (....., 3) in old basis
    basis ... new basis tensor of shape (3, 3)'''
    M = np.zeros_like(basis)
    M[0] = np.cross(basis[1], basis[2])
    M[1] = np.cross(basis[2], basis[0])
    M[2] = np.cross(basis[0], basis[1])
    V = np.dot(basis[0], np.cross(basis[1], basis[2]))

    #print(np.dot(np.cross(basis[0], basis[1]),basis[2]))
    return 1 / V * np.tensordot( v, M, axes =(-1, 1))

