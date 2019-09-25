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
import tempfile


def get_frame_traj_and_mom(TRAJ, MOMS, n_atoms, n_moms): #by Arne Scherrer
    """iterates over TRAJECTORY and MOMENTS files and yields generator of positions, velocities and moments (in a.u.)
       script originally by Arne Scherrer
"""
    with open(TRAJ, 'r') as traj_f, open(MOMS, 'r') as moms_f:
        traj_it = (list(map(float,line.strip().split()[1:])) for line in traj_f if 'NEW DATA' not in line)
        moms_it = (list(map(float,line.strip().split()[1:])) for line in moms_f if 'NEW DATA' not in line)
        try:
            while traj_it and moms_it:
                pos_au, vel_au = tuple(np.array([next(traj_it) for i_atom in range(n_atoms)]).reshape((n_atoms, 2, 3)).swapaxes(0,1))
                wc_au, c_au, m_au = tuple(np.array([next(moms_it) for i_mom in range(n_moms)]).reshape((n_moms, 3, 3)).swapaxes(0,1))
                yield pos_au, vel_au, wc_au, c_au, m_au
        except StopIteration:
            pass


def extract_mtm_data_tmp(MTM_DATA_E0,MTM_DATA_R1,n_frames,n_states): #temporary version for debugging MTM. Demands CPMD3 output file.
    fn_buf1 = tempfile.TemporaryFile(dir='/tmp/')
    fn_buf2 = tempfile.TemporaryFile(dir='/tmp/')
    
    buf1 = np.memmap(fn_buf1,dtype='float64',mode='w+',shape=(n_frames*n_states*n_states))
    with open(MTM_DATA_E0, 'r') as f:
        for i,line in enumerate(f): buf1[i]=float(line.strip().split()[-1])

    buf2 = np.memmap(fn_buf2,dtype='float64',mode='w+',shape=(n_frames*n_states*n_states,3))             
    with open(MTM_DATA_R1, 'r') as f:
        for i,line in enumerate(f): buf2[i] = np.array(line.strip().split()[-3:]).astype(float)
    
    E0 = buf1.reshape((n_frames,n_states,n_states))                    
    R1 = buf2.reshape((n_frames,n_states,n_states,3)) #mode has to be 'MD' !
    
    del buf1, buf2
    
#Factor 2 already in CPMD --> change routine later
    return E0/2,R1/2

