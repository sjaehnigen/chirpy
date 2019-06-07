#!/usr/bin/env python

import numpy as np
import copy
from topology.symmetry import get_distance_matrix, distance_pbc, wrap, get_cell_vec
from mathematics.algebra import change_euclidean_basis as ceb

#old
from lib import constants #see ~/pythonbase

def dist_crit_aa(symbols):
    '''Get distance criteria matrix of symbols (in angstrom)'''
    natoms = len(symbols)
    crit_aa = np.zeros((natoms, natoms))
    _r = np.array([constants.species[s]['RVDW'] for s in symbols]).astype(float) / 100.0
    crit_aa = (_r[:, None] + _r[None, :])
    crit_aa *= 0.6 #http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html
    return crit_aa


##TMP solution
#def define_molecules_XYZclass(xyz):
#    '''expects one System object. I use frame 0 of mol as reference.'''
#    d = xyz
#    crit_aa = dist_crit_aa(d.symbols)
#    dist_array = d.data[0, :, None, :] - d.data[0, None, :, :] #frame 0
#    #replace by pos
##    if hasattr(mol,'UnitCell'):
##        abc = mol.UnitCell.abc
##        dist_array -= np.around(dist_array/abc[None,None])*abc[None,None]
#
#    dist_array = np.linalg.norm(dist_array, axis=-1)
#    dist_array[dist_array == 0.0] = 'Inf'
#
#    neigh_map = np.array([(dist_array <= crit_aa)])[0].astype(bool)
#    h, noh = np.array([d.symbols == 'H'])[0], np.array([d.symbols != 'H'])[0]
#    n_noh = noh.sum()
#
#    n_mol = 0
#    fragment = np.zeros((n_noh))
#    atom_count = n_noh
#
#    for atom in range(n_noh):
#        if fragment[atom] == 0:
#            n_mol += 1
#            fragment, atom_count = assign_molecule(
#                fragment,
#                n_mol,
#                n_noh,
#                neigh_map[noh][:, noh],
#                atom,
#                atom_count
#                )
#        if atom_count == 0:
#            break
#
#    ass = np.zeros((d.n_atoms)).astype(int)
#    ass[noh] = fragment
#    ass[h] = ass[np.argmin(dist_array[h], axis=1)]
#    return ass

def _make_batches(p, nb, cell_aa_deg=None, ov=None):
    '''nb .. number of batches: tuple (nx, ny. nz)'''
    if cell_aa_deg is not None:
        MAX = cell_aa_deg[:3]
        MIN = np.zeros_like(MAX)
    else:
        MIN = tuple([ np.amin(_ip) for _ip in np.moveaxis(p, -1, 0) ])
        MAX = tuple([ np.amax(_ip) for _ip in np.moveaxis(p, -1, 0) ])
    X = np.array([np.linspace(_min, _max, _n + 1) for _min, _max, _n in zip(MIN,MAX,nb)])
    if ov == None:
        ov = [(_x[1] - _x[0]) / 4. for _x in X]
        print(X)
        print(ov)
    Y = np.array([
        [(_x0-_ov, _x1+_ov) for _x0, _x1 in zip(_x, np.roll(_x, -1))][:-1] for _x, _ov in \
        zip(X, ov) ])
    Y0 = np.array(np.meshgrid(*[np.array(_y)[:,0] for _y in Y])).reshape((3,-1)).T
    Y1 = np.array(np.meshgrid(*[np.array(_y)[:,1] for _y in Y])).reshape((3,-1)).T
    Z = np.array([(_y0, _y1) for _y0, _y1 in zip(Y0, Y1)])

    return Z


#ToDo remove class dependence
def define_molecules(mol):
    '''expects one System(Molecule, Supercell, ...) object. I use frame 0 of mol as reference.'''
    d = mol.XYZData
    cell_aa_deg = getattr(mol,"cell_aa_deg")

    if d._type == "trajectory":
        _p = d.pos_aa[0]
    elif d._type == "frame":
        _p = d.pos_aa
    h, noh = np.array([d.symbols == 'H'])[0], np.array([d.symbols != 'H'])[0]
    n_mol = 0
    fragment = np.zeros((noh.sum()))
    ass = np.zeros((d.n_atoms)).astype(int)

    # This is now using explicit batch edges, more elegant would it be to use implicit edges that
    # have a starting point (minimum) and interval size (however it does not allow for more
    # sophisticated adaptive batch generation (e.g. using different sizes).

    # change basis
    if cell_aa_deg is not None:
        cell_vec_aa = get_cell_vec(cell_aa_deg)
        _p = ceb(_p, cell_vec_aa)
        _cell = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
    def _w(p):
        try:
            return wrap(p, _cell)
        except TypeError:
            return p

    # However, it does not work yet for _n_b != (1,1,1) because too many molecules are found in each
    # batch (NB: molecular fragments will also get a mol number). For old code see below.
    _n_b = (1,1,1)

    # NB: do not wrap batch positions!
    # What is the influence of ov?
    #_batch = _make_batches(_p, _n_b, cell_aa_deg=_cell, ov=2.0/np.linalg.norm(cell_aa_deg[:3]))
    #_batch = _make_batches(_p, _n_b, cell_aa_deg=_cell, ov=None) #2.0/np.linalg.norm(cell_aa_deg[:3]))
    _batch = _make_batches(_p, _n_b, cell_aa_deg=_cell, ov=[0.0, 0.0, 0.0]) #2.0/np.linalg.norm(cell_aa_deg[:3]))
    for _ib, _b in enumerate(_batch):
        _ind = np.prod(
                _w(_p - _b[0, None]) <= _b[1, None] - _b[0, None], axis=-1
                ).astype(bool)
        _pp = _p[_ind,:]

        #return to basis
        if cell_aa_deg is not None:
            _pp = np.tensordot(_pp, cell_vec_aa, axes=1)
        dist_array = get_distance_matrix(_pp, cell_aa_deg=cell_aa_deg)
        dist_array[dist_array == 0.0] = 'Inf'
        crit_aa = dist_crit_aa(d.symbols[_ind])

        neigh_map = dist_array <= crit_aa

        _ind2 = _ind[noh]
        n_noh = _ind2.sum()

        atom_count = n_noh
        # sort zeros to back of the array
        _sort = np.roll(np.argsort(fragment[_ind2]),-(fragment[_ind2]==0).sum())
        _mol = fragment[_ind2][_sort]
        _neigh = neigh_map[noh[_ind]][:,noh[_ind]]
        _neigh = _neigh[_sort][:,_sort]
        for atom in range(n_noh):
            if _mol[atom] == 0:
                n_mol += 1
                _mol, atom_count = assign_molecule(_mol, n_mol, n_noh, _neigh, atom, atom_count)
            if atom_count == 0:
                break
        # This is critical: somehow the connection between batches has to be found. At the moment
        # molecular fragments are not joined

        # Basic idea ----------------------------
        def _store_mol(f,m):
        #    new_f = []
        #    for _f, _m in zip(f,m):
        #        if _f != 0:
        #            m[m==_m] = _f #does not work(m is not updated)?
        #        else:
        #            _f = _m
        #        new_f.append(_f)
        #    return np.array(new_f)
        # OR
            for _if, _f in enumerate(f):
                if _f != 0:
                    if m[_if] != 0:
                        m[m == m[_if]] = _f
                    else:
                        m[_if] = _f
            return m
        # BOTH DO NOT WORK....
        # How to treat cross-batch connections correctly?
        # ---------------------------------------
        if sum(_n_b) != 3:
            fragment[_ind2] = _store_mol(fragment[_ind2], _mol[np.argsort(_sort)])
            n_mol = np.amax(fragment)
        else:
            # Revert sorting
            fragment[_ind2] = _mol[np.argsort(_sort)]


        # This should be done at the end (outside the loop), but needs recalculation of dist_array
        ass[noh] = fragment
        ass[(_ind*h).astype(bool)] = ass[np.argmin(dist_array[h[_ind]], axis=1)]
        #if sum(_n_b) != 3:
        #    print(ass)
    return ass

# OLD WORKING CODE backup
#    if cell_aa_deg is not None:
#        _p = np.tensordot(_p, cell_vec_aa, axes=1)
#    print( np.allclose(_p,_p0 ))
#    dist_array = get_distance_matrix(_p, cell_aa_deg=cell_aa_deg)
#    dist_array[dist_array == 0.0] = 'Inf'
#
#    neigh_map = dist_array <= crit_aa
#    h, noh = np.array([d.symbols == 'H'])[0], np.array([d.symbols != 'H'])[0]
#    n_noh = noh.sum()
#
#    n_mol = 0
#    fragment = np.zeros((n_noh))
#    atom_count = n_noh
#
#    for atom in range(n_noh):
#        if fragment[atom] == 0:
#            n_mol += 1
#            fragment, atom_count = assign_molecule(
#                fragment,
#                n_mol,
#                n_noh,
#                neigh_map[noh][:, noh],
#                atom,
#                atom_count
#                )
#        if atom_count == 0:
#            break
#
#    ass = np.zeros((d.n_atoms)).astype(int)
#    ass[noh] = fragment
#    ass[h] = ass[np.argmin(dist_array[h], axis=1)]
##    print(np.array([np.argwhere(_n).ravel().tolist() for _n in neigh_map]))
##    conn = [np.argwhere(_n).ravel().tolist() for _n in neigh_map]
#    return ass

def assign_molecule(molecule, n_mol, n_atoms, neigh_map, atom, atom_count):
    '''This method can do more than molecules! See BoxObject
    molecule … assignment
    n_mol … species counter
    n_atoms … total number of entries
    neigh_map … partner matrix
    atom … current line in partner matrix
    atom_count … starts with n_atoms until zero
    '''
    molecule[atom] = n_mol
    atom_count -= 1
    for i in range(n_atoms):
        if neigh_map[atom, i] and molecule[i] == 0:
            molecule, atom_count = assign_molecule(
                molecule,
                n_mol,
                n_atoms,
                neigh_map,
                i,
                atom_count
                )
        if atom_count == 0:
            break
    return molecule, atom_count

#EOF
