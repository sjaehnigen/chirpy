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

def magnetic_dipole_shift_origin(pos_au,c_au,origin_au=np.array([0.0,0.0,0.0]),cell_au=None):
    v_trans_au = pos_au-origin_au[np.newaxis,:]
    if hasattr(cell_au,'shape'): #check for np array, not so pretty
        v_trans_au -= np.around(v_trans_au/cell_au)*cell_au
    return 0.5*np.sum(eijk[np.newaxis,:,:,:]*v_trans_au[:,:,np.newaxis,np.newaxis]*c_au[:,np.newaxis,:,np.newaxis], axis=(0,1,2))



#def magnetic_dipole_moment_shift_origin_pbc(m_old_au,pos_au,origin_au=np.array([0.0,0.0,0.0])):

