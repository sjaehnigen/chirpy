#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np
from ..topology.mapping import dist_crit_aa, get_distance_matrix, distance_pbc, wrap, get_cell_vec
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..read.coordinates import pdbReader


# NB: No reader should be called in this module


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
    Y = np.array([
        [(_x0-_ov, _x1+_ov) for _x0, _x1 in zip(_x, np.roll(_x, -1))][:-1] for _x, _ov in \
        zip(X, ov) ])
    Y0 = np.array(np.meshgrid(*[np.array(_y)[:,0] for _y in Y])).reshape((3,-1)).T
    Y1 = np.array(np.meshgrid(*[np.array(_y)[:,1] for _y in Y])).reshape((3,-1)).T
    Z = np.array([(_y0, _y1) for _y0, _y1 in zip(Y0, Y1)])

    return Z


# ToDo remove class dependence
def define_molecules(mol):
    '''expects one System(Molecule, Supercell, ...) object. I use frame 0 of mol as reference.'''
    d = mol.XYZ
    symbols = np.array(d.symbols)
    cell_aa_deg = getattr(mol,"cell_aa_deg")

    if d._type == "trajectory":
        _p = d.pos_aa[0]
    elif d._type == "frame":
        _p = d.pos_aa
    h, noh = np.array([symbols == 'H'])[0], np.array([symbols != 'H'])[0]
    n_noh = noh.sum()
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

    # However, it does not work yet for large systems (deadlock?) 
    # ToDo: automatic batch definition
    _n_b = (4, 4, 4)

    # NB: do not wrap batch positions!
    _batch = _make_batches(_p, _n_b, cell_aa_deg=_cell, ov=3*[2.0/np.linalg.norm(cell_aa_deg[:3])])
    #_batch = _make_batches(_p, _n_b, cell_aa_deg=_cell) #dynamic ov may lead to problems for small
    #batches
    neigh_list = []
    for _ib, _b in enumerate(_batch):
        _ind = np.prod(
                _w(_p - _b[0, None]) <= _b[1, None] - _b[0, None], axis=-1
                ).astype(bool)
        _ind2 = np.argwhere(_ind[noh]).T[0]
        _pp = _p[_ind,:]

        #return to basis
        if cell_aa_deg is not None:
            _pp = np.tensordot(_pp, cell_vec_aa, axes=1)

        dist_array = get_distance_matrix(_pp, cell_aa_deg=cell_aa_deg)
        dist_array[dist_array == 0.0] = 'Inf'
        crit_aa = dist_crit_aa(symbols[_ind])

        neigh_map = dist_array <= crit_aa
        neigh_list += [tuple(_ind2[_l]) for _l in np.argwhere(neigh_map[noh[_ind]][:, noh[_ind]]==1)]

    #---> until here it is fast
    #neigh_list = [tuple(_i) for _i in np.sort(np.unique(np.sort(neigh_list, axis=-1), axis=0), axis=0)]
    neigh_list = [tuple(_i) for _i in np.unique(np.sort(neigh_list, axis=-1), axis=0)]
    #print(neigh_list[:10])
    if cell_aa_deg is not None:
        _p = np.tensordot(_p, cell_vec_aa, axes=1)

    #OLD (unbatched)
    #dist_array = get_distance_matrix(_p, cell_aa_deg=cell_aa_deg)
    #dist_array[dist_array == 0.0] = 'Inf'
    #crit_aa = dist_crit_aa(d.symbols)

    #neigh_map = dist_array <= crit_aa
    #neigh_list = [tuple(_l) for _l in np.argwhere(neigh_map[noh][:, noh]==1)]
    def neigh_function(a,i):
        #faster?
        if tuple(sorted((a, i))) in neigh_list:# or (i, a) in neigh_list:
        #if (a, i) in neigh_list or (i, a) in neigh_list:
            return True
        else:
            return False

    n_mol = 0
    fragment = np.zeros((n_noh))
    atom_count = n_noh

    #for atom in range(n_noh):
    #    if fragment[atom] == 0:
    #        n_mol += 1
    #        fragment, atom_count = assign_molecule(
    #            fragment,
    #            n_mol,
    #            n_noh,
    #            neigh_map[noh][:, noh],
    #            atom,
    #            atom_count
    #            )
    #    if atom_count == 0:
    #        break

    # ---> the nesting is slow and ineffcient
    # Reason: Checking neigh_list expensive?
    # It also gives wrong results sometimes
    for atom in range(n_noh):
        if fragment[atom] == 0:
            n_mol += 1
            #print(n_mol)
            fragment, atom_count = assign_molecule_NEW(
                fragment,
                n_mol,
                n_noh,
                neigh_function,
                atom,
                atom_count
                )
        if atom_count == 0:
            break

    ass = np.zeros((d.n_atoms)).astype(int)
    ass[noh] = fragment
    # This is more complicated (and expensive), but is batch-compatible as it avoids accessing
    # dist_array
    for _h in np.argwhere(h):
        _d = np.linalg.norm(distance_pbc(_p[_h], _p, cell_aa_deg=cell_aa_deg), axis=-1)
        _d[_d == 0.0] = 'Inf'
        _i = np.argmin(_d)
        ass[_h] = ass[_i]

    # Old
    #ass[h] = ass[np.argmin(dist_array[h], axis=1)]

    # return molecular centres of mass (and optionally wrap mols?)

    return ass

def assign_molecule_NEW(molecule, n_mol, n_atoms, neigh_map, atom, atom_count):
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
        if neigh_map(atom, i) and molecule[i] == 0:
            molecule, atom_count = assign_molecule_NEW(
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


def read_topology_file(fn, **kwargs):
    '''Only PDB support for now'''
    fmt = fn.split('.')[-1]
    if fmt == 'pdb':
        # A little old messy code
        data, types, symbols, residues, cell_aa_deg, title = pdbReader(fn)
        residues = np.array(residues).astype(str)
        resi = ['-'.join(_r) for _r in residues]
        # python>=3.6: keeps order
        _map_dict = dict(zip(list(dict.fromkeys(resi)), range(len(set(resi)))))
        mol_map = [_map_dict[_r] for _r in resi]
        n_mols = len(set(mol_map))  # max(mol_map)+1
        # n_map, symbols = zip(*[(im, symbols[ia])
        #                        for ia, a in enumerate(n_map) for im, m in
        #                        enumerate(kwargs.get('extract_mols',
        #                                             range(n_mols))) if a==m])
        if n_mols != max(mol_map)+1:
            raise ValueError('STH is wrong')

        return {'mol_map': mol_map,
                'symbols': symbols,
                'cell_aa_deg': cell_aa_deg}

    else:
        raise ValueError('Unknown format: %s.' % fmt)
