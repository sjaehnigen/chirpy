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
from ..topology.mapping import dist_crit_aa, distance_matrix, distance_pbc, \
        wrap, get_cell_vec, detect_lattice
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..read.coordinates import pdbReader


def fermi_cutoff_function(distance, R_cutoff, D):
    return 1 / (1 + np.exp((distance - R_cutoff) / D))


def _make_batches(MIN, MAX, nb, ov=None):
    '''Split the cartesian space into subsegments (batches).
       nb ... number of batches: tuple (nx, ny, nz)
       ov ... overlap of batches (for cross communication)
       Future: allow manual definition of batches (adaptive grid)
       '''
    _nodes = np.array([np.linspace(_min, _max, _n + 1)
                       for _min, _max, _n in zip(MIN, MAX, nb)])
    if ov is None:
        ov = [(_x[1] - _x[0]) / 3. for _x in _nodes]

    _edges = np.array([
           [(_x0-_ov, _x1+_ov) for _x0, _x1 in zip(_x, np.roll(_x, -1))][:-1]
           for _x, _ov in zip(_nodes, ov)
           ])

    _starts = np.array(np.meshgrid(
        *[np.array(_y)[:, 0] for _y in _edges])).reshape((3, -1)).T
    _ends = np.array(np.meshgrid(
        *[np.array(_y)[:, 1] for _y in _edges])).reshape((3, -1)).T

    _batches = np.array([(_y0, _y1) for _y0, _y1 in zip(_starts, _ends)])

    return _batches


def define_molecules(pos_aa, symbols, **kwargs):
    '''Expects one System (Molecule, Supercell, ...) object.
       Uses frame 0 of mol as reference.'''

    _p = pos_aa
    symbols = np.array(symbols)
    cell_aa_deg = kwargs.get("cell_aa_deg")
    n_atoms = len(symbols)

    ass = np.zeros((n_atoms)).astype(int)

    h, noh = np.array([symbols == 'H'])[0], np.array([symbols != 'H'])[0]
    n_noh = noh.sum()
    n_mol = 0
    fragment = np.zeros((noh.sum()))

    if detect_lattice(cell_aa_deg) is not None:
        # --- change basis
        cell_vec_aa = get_cell_vec(cell_aa_deg)
        _p = ceb(_p, cell_vec_aa)
        _cell = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
        MAX = _cell[:3]
        MIN = np.zeros_like(MAX)
        _n_b = tuple((cell_aa_deg[:3] / 12).astype(int) + 1)
    else:
        _cell = None
        MIN = tuple([np.amin(_ip) for _ip in np.moveaxis(_p, -1, 0)])
        MAX = tuple([np.amax(_ip) for _ip in np.moveaxis(_p, -1, 0)])
        _n_b = tuple([int((_max - _min) / 12) + 1
                      for _max, _min in zip(MAX, MIN)])

    # NB: do not wrap batch positions!
    _batch = _make_batches(MIN, MAX, _n_b)

    def _w(p):
        return wrap(p, _cell)

    neigh_list = []
    for _ib, _b in enumerate(_batch):
        _ind = np.prod(
                _w(_p - _b[0, None]) <= _b[1, None] - _b[0, None], axis=-1
                ).astype(bool)
        _ind2 = np.argwhere(_ind[noh]).T[0]
        _pp = _p[_ind, :]

        # --- return to original basis
        if detect_lattice(cell_aa_deg) is not None:
            _pp = np.tensordot(_pp, cell_vec_aa, axes=1)

        dist_array = distance_matrix(_pp, cell_aa_deg=cell_aa_deg)
        dist_array[dist_array == 0.0] = 'Inf'
        crit_aa = dist_crit_aa(symbols[_ind])

        neigh_map = dist_array <= crit_aa
        neigh_list += [tuple(_ind2[_l]) for _l in np.argwhere(
                           neigh_map[noh[_ind]][:, noh[_ind]] == 1)]

    neigh_list = [tuple(_i) for _i in np.unique(np.sort(neigh_list, axis=-1),
                                                axis=0)]

    neigh_dict = {}
    for v, k in neigh_list:
        if v not in neigh_dict:
            neigh_dict[v] = [k]
        else:
            neigh_dict[v].append(k)

    if detect_lattice(cell_aa_deg) is not None:
        _p = np.tensordot(_p, cell_vec_aa, axes=1)

    # --- ToDo: Still slow for large systems (but better with neigh_dict)
    #           Sorted data can lead to errors in neigh_list
    # def neigh_function(a, i):
    #     if tuple(sorted((a, i))) in neigh_list:
    #         return True
    #     else:
    #         return False
    def neigh_function(a, i):
        try:
            if i in neigh_dict[a] or a in neigh_dict[i]:
                return True
            else:
                return False
        except KeyError:  # Arggh
            return False

    n_mol = 0
    fragment = np.zeros((n_noh))
    atom_count = n_noh

    for atom in range(n_noh):
        if fragment[atom] == 0:
            n_mol += 1
            fragment, atom_count = assign_molecule(
                fragment,
                n_mol,
                n_noh,
                neigh_function,
                atom,
                atom_count
                )
        if atom_count == 0:
            break

    ass = np.zeros((n_atoms)).astype(int)
    ass[noh] = fragment

    # --- avoids accessing dist_array
    for _h in np.argwhere(h):
        _d = np.linalg.norm(distance_pbc(_p[_h],
                                         _p,
                                         cell_aa_deg=cell_aa_deg), axis=-1)
        _d[_d == 0.0] = 'Inf'
        _i = np.argmin(_d)
        ass[_h] = ass[_i]

    # return molecular centres of mass (and optionally wrap mols?)
    return ass


def assign_molecule(molecule, n_mol, n_atoms, neigh_map, atom, atom_count):
    '''This method can do more than molecules! See BoxObject
    molecule … assignment
    n_mol … species counter
    n_atoms … total number of entries
    neigh_map … partner matrix FUNCTION (!)
    atom … current line in partner matrix
    atom_count … starts with n_atoms until zero
    '''
    molecule[atom] = n_mol
    atom_count -= 1
    # for i in range(n_atoms):
    for i in np.argwhere(molecule == 0)[:, 0]:
        if neigh_map(atom, i):  # and molecule[i] == 0:
            molecule, atom_count = assign_molecule(
                molecule,
                n_mol,
                n_atoms,
                neigh_map,
                i,
                atom_count
                )
            # print(atom_count)
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
