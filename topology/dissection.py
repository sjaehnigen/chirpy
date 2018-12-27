#!/usr/bin/python3

import sys
import numpy as np
from lib import constants #see ~/pythonbase

def dist_crit_aa(symbols):
    natoms = len(symbols)
    crit_aa = np.zeros((natoms,natoms))
    r = np.array([constants.species[s]['RVDW'] for s in symbols]).astype(float)
    r[np.array(symbols) == 'H'] *= 1.0
    crit_aa = (r[:,np.newaxis] + r[np.newaxis,:])/100.0
    crit_aa *= (1.0/2.0)
    return crit_aa


#TMP solution
def define_molecules_XYZclass(xyz):
    '''expects one Molecule object. I use frame 0 of mol as reference.'''
    d = xyz
    crit_aa = dist_crit_aa(d.symbols)    
    dist_array = d.data[0,:,None,:]-d.data[0,None,:,:] #frame 0
    #replace by pos
#    if hasattr(mol,'UnitCell'):
#        abc = mol.UnitCell.abc
#        dist_array -= np.around(dist_array/abc[None,None])*abc[None,None]

    dist_array = np.linalg.norm(dist_array,axis=-1)
    dist_array[dist_array == 0.0] = 'Inf'

    neigh_map  = np.array([(dist_array <= crit_aa)])[0].astype(bool) 
    h,noh    = np.array([d.symbols == 'H'])[0],np.array([d.symbols != 'H'])[0]
    n_noh  = noh.sum()

    n_mol      = 0
    fragment   = np.zeros((n_noh))
    atom_count = n_noh

    for atom in range(n_noh):
        if fragment[atom] == 0:
            n_mol += 1       
            fragment,atom_count = assign_molecule(fragment,n_mol,n_noh,neigh_map[noh][:,noh],atom,atom_count)
        if atom_count == 0:
            break

    ass      = np.zeros((d.n_atoms)).astype(int)   
    ass[noh] = fragment
    ass[h]   = ass[np.argmin(dist_array[h],axis=1)]
    return ass

# ToDo remove class dependence
def define_molecules(mol):
    '''expects one Molecule object. I use frame 0 of mol as reference.'''
    d = mol.XYZData
    crit_aa = dist_crit_aa(d.symbols)    
    dist_array = d.data[0,:,None,:]-d.data[0,None,:,:] #frame 0
    #replace by pos
    if hasattr(mol,'UnitCell'):
        abc = mol.UnitCell.abc
        dist_array -= np.around(dist_array/abc[None,None])*abc[None,None]

    dist_array = np.linalg.norm(dist_array,axis=-1)
    dist_array[dist_array == 0.0] = 'Inf'

    neigh_map  = np.array([(dist_array <= crit_aa)])[0].astype(bool) 
    h,noh    = np.array([d.symbols == 'H'])[0],np.array([d.symbols != 'H'])[0]
    n_noh  = noh.sum()

    n_mol      = 0
    fragment   = np.zeros((n_noh))
    atom_count = n_noh

    for atom in range(n_noh):
        if fragment[atom] == 0:
            n_mol += 1       
            fragment,atom_count = assign_molecule(fragment,n_mol,n_noh,neigh_map[noh][:,noh],atom,atom_count)
        if atom_count == 0:
            break

    ass      = np.zeros((d.n_atoms)).astype(int)   
    ass[noh] = fragment
    ass[h]   = ass[np.argmin(dist_array[h],axis=1)]
    return ass

def assign_molecule(molecule,n_mol,n_atoms,neigh_map,atom,atom_count):
    molecule[atom] = n_mol
    atom_count -= 1
    for i in range(n_atoms):
        if neigh_map[atom,i] and molecule[i] == 0:
            molecule,atom_count = assign_molecule(molecule,n_mol,n_atoms,neigh_map,i,atom_count)
        if atom_count == 0:
            break
    return molecule,atom_count

#NB: the molecules have to be sequentially numbered starting with 0, the script will transform them starting with 0
def dec(prop, indices):
    """decompose prop according to indices"""
    return [np.array([prop[k] for k, j_mol in enumerate(indices) if j_mol == i_mol]) for i_mol in range(max(indices)+1)]









#EOF


