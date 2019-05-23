#!/usr/bin/python3

import numpy as np
import sys
import copy


def get_atom_spread( pos ):
    return np.array( [ np.amax( _p ) - np.amin( _p ) for _p in np.moveaxis( pos, -1, 0 ) ] )

def map_atoms_by_coordinates(mol1,mol2):
    '''expects two Molecule objects of the same shape. I use frame 0 of mol2 as reference. Not tested for long trajectories of mol1.'''
#pos_aa,sym,box_aa,replica,vec_trans,pos_cryst,sym_cr):
    d1 = mol1.XYZData
    d2 = mol2.XYZData
    ie,tmp = d1._is_similar(d2)
    if not ie:
        print('ERROR: The two Molecule objects are not similar!\n n_atoms: %s\n n_fields: %s\n symbols: %s'%tuple(tmp))
        sys.exit(1)
#    if hasattr(mol1,'UnitCell'):
#        cell_aa = mol1.UnitCell.abc #only albega = 90 degreee for now

    def get_assign(pos1,pos2,unit_cell=None):
        dist_array = pos1[:,:,None,:]-pos2[0,None,None,:,:]
        if unit_cell:
            dist_array-= np.around(dist_array/unit_cell.abc)*unit_cell.abc
        return np.argmin(np.linalg.norm(dist_array,axis=-1),axis=2)
        #insert routine to avoid double index
    
    assign=np.zeros((d1.n_frames,d1.n_atoms)).astype(int)
    for s in np.unique(d1.symbols):
        i1 = d1.symbols==s
        i2 = d2.symbols==s
        ass = get_assign(d1.data[:,i1,:3],d2.data[:,i2,:3],unit_cell=getattr(mol1,'UnitCell',None))
        assign[:,i1]=np.array([np.arange(d2.n_atoms)[i2][ass[fr]] for fr in range(d1.n_frames)]) #maybe unfortunate to long trajectories
    return assign

def kabsch_algorithm(P,ref):
    C = np.dot(np.transpose(ref), P)
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
    # Create Rotation matrix U
    return np.dot(V, W) 

def align_atoms(pos_mobile,w,**kwargs): #demands a reference for each frame, respectively 
    pos_mob = copy.deepcopy(pos_mobile)
    pos_ref = kwargs.get('ref',pos_mobile[0])
    pos_ref -= (np.sum(pos_ref*w[:,None], axis=-2)/w.sum())[None,:]
    com = np.sum(pos_mob*w[None,:,None], axis=-2)/w.sum()
    pos_mob -= com[:,None,:]

    for frame,P in enumerate(pos_mob):
        U = kabsch_algorithm(P*w[:,None],pos_ref*w[:,None])
        pos_mob[frame] = np.tensordot(U,P,axes=([1],[1])).swapaxes(0,1)
            #V = np.tensordot(U,P,axes=([1],[1])).swapaxes(0,1) #velocities (not debugged)
    pos_mob += com[:,None,:]
    return pos_mob
