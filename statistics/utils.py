#!/usr/bin/env python3

#import sys 
#import os
import numpy as np

#migrate these tools to geometry or so
def dihedral(p1,p2,p3,p4):
    '''Definition along given points p1--p2--p3--p4; all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    v1 = p1-p2
    v2 = p3-p2
    v3 = p4-p3
    n1 = np.cross(v1,v2)
    n2 = np.cross(v3,v2)
    dr = np.inner(np.cross(n1,n2),v2)
    dr /= np.linalg.norm(dr)
    #dih = np.arccos(np.inner(n1,n2)/(np.linalg.norm(n1,axis=-1)*np.linalg.norm(n2,axis=-1)))[0]
    dih = np.arccos(np.inner(n1,n2)/(np.linalg.norm(n1,axis=-1)*np.linalg.norm(n2,axis=-1)))
    dih*=dr #Direction of rotation
    return dih

