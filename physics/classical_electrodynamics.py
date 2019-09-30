#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


import numpy as np
from ..physics import constants, kspace

eijk = np.zeros((3, 3, 3))
eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = +1
eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1

#m magnetic dipole moment
#c current ipole moment

def _get_divrot(data, cell_au):
    """data of shape 3, x, y, z"""
    gradients = np.array(np.gradient(data, 1, cell_au[0][0], cell_au[1][1], cell_au[2][2])[1:])
    div = gradients.trace(axis1=0, axis2=1)
    rot = np.einsum('ijk,jklmn->ilmn', eijk, gradients)
    return div, rot

def current_dipole_moment(vel_au,charges):
    return vel_au*charges[:,np.newaxis]

#def magnetic_dipole_shift_origin(pos_au,c_au,origin_au=np.array([0.0,0.0,0.0]),cell_au=None):
def magnetic_dipole_shift_origin_OLD(pos_au,c_au,origin_au,**kwargs):
    cell_au = kwargs.get('cell_au')
    v_trans_au = pos_au-origin_au[None,:]
    if hasattr(cell_au,'shape'): #check for np array, not so pretty
        v_trans_au -= np.around(v_trans_au/cell_au)*cell_au
    return 0.5*np.sum(eijk[None,:,:,:]*v_trans_au[:,:,None,None]*c_au[:,None,:,None], axis=(0,1,2))

def magnetic_dipole_shift_origin(c_au,trans_au,**kwargs):
    if len(c_au.shape)==2:
        return 0.5*np.sum(eijk[None,:,:,:]*trans_au[:,:,None,None]*c_au[:,None,:,None], axis=(1,2))#axis 0?
    if len(c_au.shape)==3:
        return 0.5*np.sum(eijk[None,None,:,:,:]*trans_au[:,:,:,None,None]*c_au[:,:,None,:,None], axis=(2,3))#sum over mols (axis 1) done later

#def magnetic_dipole_moment_shift_origin_pbc(m_old_au,pos_au,origin_au=np.array([0.0,0.0,0.0])):
def coulomb(r0, r, q, thresh=1.E-8):
    '''r...shape(N,...,3)'''
    d = r0 - r
    d3 = np.linalg.norm(d, axis=-1)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3<thresh**3, 0.0, np.divide(1.0, d3))
    E = q * d * d3_inv[:,None]
    return E

def coulomb_grid(r, rho, pos_grid, voxel, thresh=1.E-8):
    '''r...shape(3,...,N)'''
    d = r[:, None, None, None]-pos_grid
    d3 = np.linalg.norm(d, axis=0)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3<thresh**3, 0.0, np.divide(1.0, d3))
    E = rho[None] * d * d3_inv[None] * voxel
    E = E.sum(axis=(1, 2, 3))
    return E

def coulomb_kspace(rho, cell_au, voxel):
    pass

def biot_savart(r0, r, j, thresh=1.E-8):
    '''r...shape(N,...,3)'''
    #in atomic units using cgs convention for B field (µ0 = 1/c)
    d = r0 - r
    d3 = np.linalg.norm(d, axis=-1)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3<thresh**3, 0.0, np.divide(1.0, d3))
    B = np.cross(j,d,axisa=-1,axisb=-1,axisc=-1) * d3_inv[:,None]
    return B / constants.c_au

def biot_savart_grid(r, j, pos_grid, voxel, thresh=1.E-8):
    '''r...shape(3,...,N)'''
    #in atomic units using cgs convention for B field (µ0 = 1/c)
    d = r[:, None, None, None]-pos_grid
    d3 = np.linalg.norm(d, axis=0)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3<thresh**3, 0.0, np.divide(1.0, d3))
    B = np.cross(j,d,axisa=0,axisb=0,axisc=0) * d3_inv[None] * voxel
    B = B.sum(axis=(1, 2, 3))
    return B / constants.c_au

def biot_savart_kspace(j, cell_au, voxel):
    div,rot = _get_divrot(j, cell_au)

    #G != 0
    B1 = kspace.k_potential(rot[0], cell_au)[1]
    B2 = kspace.k_potential(rot[1], cell_au)[1]
    B3 = kspace.k_potential(rot[2], cell_au)[1]
    B = np.array([B1, B2, B3]) / (4 * np.pi) #this 4 pi division should be done in kspace binary already?

    #G == 0 ???
    return B * voxel / constants.c_au #*2??

