#!/usr/bin/env python

import numpy as np
from ..physics import constants

eijk = np.zeros((3, 3, 3))
eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = +1
eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1

#m magnetic dipole moment
#c current ipole moment

def calculate_mic(e0,r1_pert,sw_c,n_states,r_wc_aa,box_vec_aa):
    sw_m_ic_r,sw_m_ic_t = calculate_mic_rotation(e0,r1_pert,n_states,r_wc_aa,box_vec_aa), calculate_mic_translation(sw_c,n_states,r_wc_aa,box_vec_aa)
    return sw_m_ic_r.sum(axis=0).sum(axis=0),sw_m_ic_t.sum(axis=0)

def calculate_mic_rotation(e0,r1_pert,n_states,r_wc_aa,box_vec_aa):
#case 0 if states interact within box (M_LC), +1 if state in column is considered +R with respect to state in line, et VV.
    case       = np.abs(r_wc_aa[:,np.newaxis,:] -r_wc_aa[:,:])
    case       = (case/box_vec_aa[np.newaxis,np.newaxis,:]*2).astype(np.int)
    #case [case  < 0] = 0

    v  = r1_pert*e0[:,:,np.newaxis]
#    v  = 2*(data['R1_pert'] - np.swapaxes(data['R1_pert'],0,1))*data['E0'][:,:,np.newaxis]#*abs(case)
    #print(v.shape)

    sw_m_ic_r = np.zeros((n_states,n_states,3))
    #for m in range(3):
    #    for o in range(3):
    #        for p in range(3):
    #            #sw_m_ic_r[:,:,m] += 2*eps[m,p,o]*v[:,:,o]*box_vec_aa[p]*case[:,:,p]*constants.l_aa2au
    #            sw_m_ic_r[:,:,m] += 2*eijk[m,p,o]*v[:,:,o]*box_vec_aa[p]*case[:,:,p]*constants.l_aa2au
    sw_m_ic_r = -2*np.sum(eijk[np.newaxis,np.newaxis,:,:,:]*box_vec_aa[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]*case[:,:,:,np.newaxis,np.newaxis]*constants.l_aa2au*v[:,:,np.newaxis,:,np.newaxis], axis=(2,3,4))

    return sw_m_ic_r/2 #DEBUG sign; I think the case has to be mirrored


def calculate_mic_translation(sw_p_pert,n_states,r_wc_aa,box_vec_aa):
#translational part
    sw_m_ic_t = np.zeros((n_states,3))
    t = sw_p_pert
#    for m in range(3):
#        for o in range(3):
#            for p in range(3):
#                #sw_m_ic_t[:,m] += 2*eps[m,p,o]*t[:,o]*box_vec_aa[p]*constants.l_aa2au
#                sw_m_ic_t[:,m] += 2*eijk[m,p,o]*t[:,o]*box_vec_aa[p]*constants.l_aa2au
    #sw_m_ic_t = 2*np.sum(eijk[np.newaxis,:,:,:]*box_vec_aa[np.newaxis,:,np.newaxis,np.newaxis]*constants.l_aa2au*t[:,np.newaxis,:,np.newaxis], axis=(1,2,3))
    sw_m_ic_t = -2*np.sum(eijk[np.newaxis,:,:,:]*box_vec_aa[np.newaxis,np.newaxis,:,np.newaxis]*constants.l_aa2au*t[:,:,np.newaxis,np.newaxis], axis=(1,2,3))

#        sw_m_ic_t   = sw_m_ic_t #- TranslateMagneticMoments(data['SW_P_VF_pert'], r_wc_aa)#-data['nuc_coc_au'][np.newaxis,:]*constants.l_au2aa)    
    return sw_m_ic_t#/2

