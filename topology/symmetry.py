#!/usr/bin/env python

import numpy as np
import copy

from topology.mapping import dec

from mathematics.algebra import change_euclidean_basis as ceb

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
    priority, and the correct cell vectors are calculated. However, VMD or other programmes may \
    still calculate wrong cell vectors. Hence, ...
    Priority should always be (0,1,2) and symmetry conventions be used (e.g. for monoclinic cells: \
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
    cell_vec_aa = get_cell_vec(cell_aa_deg)

    if not any([_a <= 0.0 for _a in cell_aa_deg[:3]]):
        return pos_aa - np.tensordot(np.floor(ceb(pos_aa, cell_vec_aa)),
                                     cell_vec_aa,
                                     axes=1
                                     )
    else:
        print( 'WARNING: Cell size zero!' )
        return pos_aa

def distance_pbc(p1, p0, **kwargs):
    _d = p1 - p0
    try:
        return _d - _pbc_shift(_d, kwargs.get("cell_aa_deg"))
    except TypeError:
        print('i am here')
        return _d

def _pbc_shift(_d, cell_aa_deg):
    '''_d of shape ...'''
    if not any([_a <= 0.0 for _a in cell_aa_deg[:3]]):
        cell_vec_aa = get_cell_vec(cell_aa_deg)
        _c = ceb(_d, cell_vec_aa)
        return np.tensordot(np.around(_c), cell_vec_aa, axes=1)
    else:
        return _d

def get_distance_matrix( pos_aa, **kwargs):
    '''pos_aa of shape (n_atoms, three) ... (FRAME)'''
    # ToDo: the following lines explode memory for many atoms ==> do coarse mapping beforehand
    # (overlapping batches) or set a max limit for n_atoms
    if pos_aa.shape[0] > 1000:
        raise MemoryError(
        'Too many atoms for molecular recognition (>1000 atom support in a future version)!'
        )
    dist_array = pos_aa[:, None, :] - pos_aa[None, :, :]
    cell_aa_deg = kwargs.get("cell_aa_deg")
    if cell_aa_deg is not None:
        dist_array -= _pbc_shift(dist_array, cell_aa_deg)

    if kwargs.get("cartesian") is not None:
        return dist_array
    else:
        return np.linalg.norm(dist_array, axis=-1)

def cowt(pos_aa, wt, **kwargs):
    '''Calculate centre of weight, consider periodic boundaries before calling this method.'''
    _p = copy.deepcopy(pos_aa) #really necessary?
    #cell_aa_deg = kwargs.get("cell_aa_deg")
    return np.sum(_p * wt[None, :, None], axis=1) / wt.sum()

def wrap_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs):
    '''DEPRECATED, use join_molecules()'''
    join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs)

def join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs): #another routine would be complete_molecules for both-sided completion
    '''pos_aa (in angstrom) with shape ( n_frames, n_atoms, three )
    Has still problems with cell-spanning molecules'''
    n_frames, n_atoms, three = pos_aa.shape
    w = kwargs.get('weights', np.ones((n_atoms)))
    w = dec(w, mol_map)

    mol_c_aa = []
    #for i_mol in range( max( mol_map ) + 1 ): #ugly ==> change it
#    def _get_nearest_neighbours(p):
#        np.argmin(get_distance_matrix(p, cell_aa_deg), axis=1)
    #------------------
    for i_mol in set(mol_map):
        ind = np.array(mol_map) == i_mol
        _p = pos_aa[:, ind]
        # reference atom: 0 (this may lead to problems, say, for large, cell-spanning molecules)
        _r = [0]*sum(ind)
        # choose as reference atom the one with smallest distances to all atoms (works better but
        # still not perfect
        _r = np.argmin(np.linalg.norm(get_distance_matrix(_p[0], cell_aa_deg=cell_aa_deg), axis=1))
        _r = [_r]*sum(ind)
        #needs an adaptive scheme (different reference points for different atoms)
        #_r = 
        _p -= _pbc_shift(_p - _p[:, _r, :], cell_aa_deg)
        # actually needs connectivity pattern to wrap everything correctly
        c_aa = cowt(_p, w[i_mol])
        mol_c_aa.append(wrap(c_aa, cell_aa_deg))
        pos_aa[:, ind] = _p - (c_aa - mol_c_aa[-1])[:, None, :]

    return pos_aa, mol_c_aa

