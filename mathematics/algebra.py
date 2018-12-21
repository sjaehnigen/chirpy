#!/usr/bin/env python3

import numpy as np

def rotation_matrix(v1,v2):
    '''rotate v1 to match v2'''
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)  
    n = np.cross(u1,u2)
    V = np.matrix([[0, -n[2], n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    R = I + V +V**2*(1-np.dot(u1,u2))/np.linalg.norm(n)**2

    return np.asarray( R ) #this matrix/array dualism may lead to trouble
