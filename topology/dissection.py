#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np
from ..topology.mapping import dist_crit_aa, distance_matrix, distance_pbc, \
        wrap, get_cell_vec, detect_lattice
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..read.coordinates import pdbReader, xyzReader


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
    '''Distance analysis in batches to create a neighbour list which is
       further evaluated to obtain clusters/molecules.
       Expects positions in angstrom of shape (n_atoms, three).
       It returns a list with assignments.'''

    _p = pos_aa
    if len(_p.shape) != 2:
        raise TypeError('Positions not in shape (n_atoms, three)!')

    symbols = np.array(symbols)
    cell_aa_deg = kwargs.get("cell_aa_deg")
    n_atoms = len(symbols)

    h = np.array([symbols == 'H'])[0]
    noh = np.array([symbols != 'H'])[0]
    n_noh = noh.sum()
    n_mol = 0
    fragment = np.zeros((noh.sum()))
    ass = np.zeros((n_atoms)).astype(int)

    _lattice = detect_lattice(cell_aa_deg)

    if _lattice is not None:
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

    # --- use batches to create pair lists and neighbour counts
    _batch = _make_batches(MIN, MAX, _n_b)

    def _w(p):
        return wrap(p, _cell)

    pair_list = []
    neigh_count = np.zeros((n_atoms))

    for _ib, _b in enumerate(_batch):
        _ind = np.prod(
                _w(_p - _b[0, None]) <= _b[1, None] - _b[0, None], axis=-1
                ).astype(bool)
        _ind2 = np.argwhere(_ind[noh]).T[0]
        _pp = _p[_ind, :]

        # --- return to original basis
        if _lattice is not None:
            _pp = np.tensordot(_pp, cell_vec_aa, axes=1)

        dist_array = distance_matrix(_pp, cell_aa_deg=cell_aa_deg)
        np.set_printoptions(precision=2, linewidth=200)
        dist_array[dist_array == 0.0] = 'Inf'
        crit_aa = dist_crit_aa(symbols[_ind])
        neigh_map = dist_array <= crit_aa

        # --- store batch info for assignment
        #     Caution: neigh_count may lead to (uncritical) overcounting if
        #              atom pairs are found in more than one batch

        neigh_count[_ind] += neigh_map.sum(axis=1)
        pair_list += [tuple(_ind2[_l]) for _l in np.argwhere(
                           neigh_map[noh[_ind]][:, noh[_ind]] == 1)]

    # --- This is a little fussy after various changes and methodology updates

    # pair_list = [tuple(_i) for _i in np.unique(np.sort(pair_list, axis=-1),
    #                                             axis=0)]

    neigh_dict = {}
    for v, k in pair_list:
        if v not in neigh_dict:
            neigh_dict[v] = [k]
        else:
            neigh_dict[v].append(k)

    neigh_list = []
    for _i in range(n_noh):
        if _i in neigh_dict:
            neigh_list.append(neigh_dict[_i])
        else:
            neigh_list.append([])

    del neigh_dict, pair_list

    if _lattice is not None:
        _p = np.tensordot(_p, cell_vec_aa, axes=1)

    n_mol = 0
    atom_count = n_noh

    for atom in np.argsort(neigh_count[noh])[::-1]:
        if fragment[atom] == 0:
            n_mol += 1
            fragment, atom_count = assign_molecule(
                fragment,
                n_mol,
                n_noh,
                neigh_list,
                atom,
                atom_count
                )
        if atom_count == 0:
            break

    ass[noh] = fragment

    # --- avoids accessing dist_array: choose closest heavy atom for H
    for _h in np.argwhere(h):
        _d = np.linalg.norm(distance_pbc(_p[_h],
                                         _p,
                                         cell_aa_deg=cell_aa_deg), axis=-1)
        _d[_d == 0.0] = 'Inf'
        _i = np.argmin(_d)
        ass[_h] = ass[_i]

    return ass


def assign_molecule(molecule, n_mol, n_atoms, neigh_map, atom, atom_count):
    '''This method can do more than molecules! See BoxObject
    molecule … assignment
    n_mol … species counter
    n_atoms … total number of entries
    neigh_map … list of neighbour atoms per atom
    atom … current line in reading neighbour map
    atom_count … starts with n_atoms until zero
    '''
    molecule[atom] = n_mol
    atom_count -= 1
    for _i in neigh_map[atom]:
        if molecule[_i] == 0:
            molecule, atom_count = assign_molecule(
                molecule,
                n_mol,
                n_atoms,
                neigh_map,
                _i,
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
            raise ValueError('Something is wrong with the list of residues!')

        return {'mol_map': mol_map,
                'symbols': symbols,
                'cell_aa_deg': cell_aa_deg}

    elif fmt == 'xyz':
        data, symbols, comments = xyzReader(fn)

        return {'symbols': symbols}

    else:
        raise ValueError('Unknown format: %s.' % fmt)
