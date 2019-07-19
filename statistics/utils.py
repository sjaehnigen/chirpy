#!/usr/bin/env python

import numpy as np

#migrate these tools to geometry or so
def angle(v1,v2):
    """VectorAngle(vec1,vec2)
Output: Angle between two vectors in deg"""
    al = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al,-1.0,1.0))
#    a_deg = np.degrees(a_rad)
    return a_rad

def signed_angle(v1,v2,n):
    """n is the reference/plane normal for angle direction"""
    al = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    a_rad = np.arccos(np.clip(al,-1.0,1.0))*np.sign(np.dot(np.cross(v1,v2),n))
    return a_rad

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
    dih_rad = np.arccos(np.inner(n1,n2)/(np.linalg.norm(n1,axis=-1)*np.linalg.norm(n2,axis=-1)))
    dih_rad*=dr #Direction of rotation
    return dih_rad

def plane_normal(p1,p2,p3):
    '''Plane spanned by vectors p1<--p2, p2-->p3; all p as 3d-np.vectors or np.arrays (last axis will be used)'''
    v1 = p1-p2
    v2 = p3-p2
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n)
    return n
