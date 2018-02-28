#!/usr/bin/python3

import numpy as np
import sys
from mgeometry.transformations import dist_crit_aa #migrate it soon

def find_methyl_groups(mol,hetatm=False):
    '''expects one Molecule object. I use frame 0 as reference. Not tested for long trajectories. Outformat is C H H H'''
#pos_aa,sym,box_aa,replica,vec_trans,pos_cryst,sym_cr):
    d = mol.XYZData
    unit_cell=getattr(mol,'UnitCell',None)
    dist_array = d.data[0,:,None,:3]-d.data[0,None,:,:3]
    if unit_cell:
        dist_array-= np.around(dist_array/unit_cell.abc)*unit_cell.abc
    dist_array = np.linalg.norm(dist_array,axis=-1)

    indh = d.symbols=='H'
    if hetatm:
        ind = d.symbols!='H'
    else:
        ind = d.symbols=='C'
    ids = np.arange(d.n_atoms)+1
    out = [[ids[i],[ids[j] for j in range(d.n_atoms) if dist_array[i,j]<dist_crit_aa(d.symbols)[i,j] and indh[j] ]] for i in range(d.n_atoms) if ind[i]]
    out = '\n'.join(['A%03d '%i[0]+'A%03d A%03d A%03d'%tuple(i[1]) for i in out if len(i[1])==3])
    print(out)
 #   def get_assign(pos1,pos2,unit_cell=None):
 #       dist_array = pos1[:,:,None,:]-pos2[0,None,None,:,:]
 #       if unit_cell:
 #           dist_array-= np.around(dist_array/unit_cell.abc)*unit_cell.abc
 #       return np.argmin(np.linalg.norm(dist_array,axis=-1),axis=2)
 #   
 #   assign=np.zeros((d1.n_frames,d1.n_atoms)).astype(int)
 #   for s in np.unique(d1.symbols):
 #       i1 = d1.symbols==s
 #       i2 = d2.symbols==s
 #       ass = get_assign(d1.data[:,i1,:3],d2.data[:,i2,:3],unit_cell=getattr(mol1,'UnitCell',None))
 #       assign[:,i1]=np.array([np.arange(d2.n_atoms)[i2][ass[fr]] for fr in range(d1.n_frames)]) #maybe unfortunate to long trajectories
 #   return assign
