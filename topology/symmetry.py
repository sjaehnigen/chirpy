#!/usr/bin/env python

import numpy as np
import sys
import copy 

from topology.dissection import dec

#old pythonbase
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


def get_cell_vec(cell_aa_deg, n_fields=3, priority=(0, 1, 2)):
    '''cell_aa_deg as np.array/list of: a b c al be ga
    n_fields usually 3
    Priority defines the alignment of non-rectangular objects in cartesian space; by convention the \
    z axis is the odd one (i.e. not aligned), and cell vectors are calculated accordingly (e.g. in \
    VMD); here, this corresponds to priority (0,1,2). 
    Any deviation (e.g. z-axis IS aligned, alpha > 90°, ...) can be taken account for by adjusting \
    priority, and the correct cell vectors are calculated. However, VMD or other programmes may not \
    still calculate the wrong cell vectors. Hence, ...
    priority should always be (0,1,2) and symmetry conventions be used (e.g. for monoclinic cells: \
    beta is the angle > 90°; CPMD wants alpha to be >90° but this is wrong and CELL VECTORS should \
    be used instead)'''

    abc, albega = cell_aa_deg[:3], cell_aa_deg[3:] * np.pi / 180.
    cell_vec_aa = np.zeros((3, n_fields))
    v0, v1, v2 = priority
    cell_vec_aa[v0, v0] = abc[v0]
    cell_vec_aa[v1, v1] = abc[v1] * np.sin(albega[(3 - v0 - v1)])
    cell_vec_aa[v1, v0] = abc[v1] * np.cos(albega[(3 - v0 - v1)])
    cell_vec_aa[v2, v2] = abc[v2] * np.sin(albega[(3 - v0 - v2)]) * np.sin(albega[(3 - v1 - v2)])
    cell_vec_aa[v2, v0] = abc[v2] * np.cos(albega[(3 - v0 - v2)])
    cell_vec_aa[v2, v1] = abc[v2] * np.cos(albega[(3 - v1 - v2)])

    return cell_vec_aa


def wrap(pos_aa, cell_aa_deg, **kwargs):
    '''pos_aa: shape (n_frames, n_atoms, three) or (n_atoms, three)
       cell: [ a b c al be ga ] (distances same dim as pos_aa; angles in degree)'''
    #--todo:
    # extend general wrap function to all symmetries
    abc, albega = np.split(cell_aa_deg, 2)

    if not np.allclose(albega, np.ones((3)) * 90.0):
        raise NotImplementedError( 'ERROR: Only orthorhombic cells implemented for mol wrap!' )

    if not any([_a <= 0.0 for _a in abc]):
        return np.remainder(pos_aa, abc[None, :])
    else:
        print( 'WARNING: Cell size zero!' )
        return pos_aa

def _cowt(pos_aa, wt, **kwargs):
    '''Calculate centre of weight, optionally within periodic boundaries.'''
    p = copy.deepcopy(pos_aa) #really necessary?
    cell_aa_deg = kwargs.get("cell_aa_deg")

    if cell_aa_deg is not None:
        abc, albega = np.split(cell_aa_deg, 2)
        if not np.allclose(albega, np.ones((3)) * 90.0):
            raise NotImplementedError( 'ERROR: Only orthorhombic cells implemented for cowt calculation!' )
        p -= np.around( (p - p[:, 0, None, :]) / abc[None, None, :]) * abc[None, None, :]
    
    return np.sum(p * wt[None, :, None], axis=1) / wt.sum()



def wrap_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs):
    '''DEPRECATED'''
    join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs)

def join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs): #another routine would be complete_molecules for both-sided completion
    '''pos_aa (in angstrom) with shape ( n_frames, n_atoms, three )'''
    n_frames, n_atoms, three = pos_aa.shape
    w = kwargs.get('weights', np.ones((n_atoms)))
    w = dec(w, mol_map)

    #abc, albega = np.split(cell_aa_deg, 2)

    mol_c_aa = []
    #cowt = lambda x,wt: np.sum(p * wt[None, :, None], axis=1) / wt.sum()

    for i_mol in range( max( mol_map ) + 1 ): #ugly ==> change it
        ind = np.array(mol_map) == i_mol
        p = pos_aa[:, ind]
    #    if not any([_a <= 0.0 for _a in abc]):
            #p -= np.around( (p - p[:, 0, None, :]) / abc[None, None, :]) * abc[None, None, :]
        c_aa = _cowt(p, w[i_mol], cell_aa_deg=cell_aa_deg)
        mol_c_aa.append(wrap(c_aa, cell_aa_deg))
            #mol_c_aa.append( np.remainder( c_aa, abc[ None, : ] ) ) #only for orthorhombic cells
    #    else:
    #        print( 'WARNING: Cell size zero!' )
    #        c_aa = _cowt(p, w[i_mol], cell_aa_deg=cell_aa_deg)
    #        mol_c_aa.append(c_aa)

        pos_aa[:, ind] = p - (c_aa - mol_c_aa[-1])[:, None, :]
        #p -= ( c_aa - mol_c_aa[ -1 ] )[ :, None, : ]

    return pos_aa, mol_c_aa
    #print('UPDATE WARNING: inserted "swapaxes(0,1)" for mol_cog_aa attribute (new shape: (n_frames,n_mols,3))!')

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
