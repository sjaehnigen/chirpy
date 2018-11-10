#!/usr/bin/python3

import numpy as np
from lib import constants #migrate to phdtools

eijk = np.zeros((3, 3, 3))
eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = +1
eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1

#m magnetic dipole moment
#c current ipole moment

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
        return 0.5*np.sum(eijk[None,:,:,:]*trans_au[:,:,None,None]*c_au[:,None,:,None], axis=(0,1,2))#axis 0?
    if len(c_au.shape)==3:
        return 0.5*np.sum(eijk[None,None,:,:,:]*trans_au[:,:,:,None,None]*c_au[:,:,None,:,None], axis=(2,3))


#def magnetic_dipole_moment_shift_origin_pbc(m_old_au,pos_au,origin_au=np.array([0.0,0.0,0.0])):

