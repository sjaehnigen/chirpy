# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2022, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import numpy as np
import copy

from ..topology.mapping import distance_pbc, wrap, cell_vec,\
    detect_lattice, neighbour_matrix, cell_l_deg
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..constants import symbols_to_symbols


def fermi_cutoff_function(distance, R_cutoff, D):
    return 1 / (1 + np.exp((distance - R_cutoff) / D))


def _make_batches(MIN, MAX, nb, ov=None):
    '''Split the cartesian space into subsegments (batches).
       nb ... number of batches: tuple (nx, ny, nz)
       ov ... overlap of batches (for cross communication)
       Future: allow manual definition of batches (adaptive grid)
       '''
    _nodes = [np.linspace(_min, _max, _n + 1)
              for _min, _max, _n in zip(MIN, MAX, nb)]
    if ov is None:
        ov = [(_x[1] - _x[0]) / 3. for _x in _nodes]

    _edges = [[(_x0-_ov, _x1+_ov)
               for _x0, _x1 in zip(_x, np.roll(_x, -1))][:-1]
              for _x, _ov in zip(_nodes, ov)]

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
        cell_vec_aa = cell_vec(cell_aa_deg)
        _p = ceb(_p, cell_vec_aa)
        _cell = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
        MAX = _cell[:3]
        MIN = np.zeros_like(MAX)
        _n_b = tuple((cell_aa_deg[:3] / 12).astype(int) + 1)
        # _n_b = tuple((cell_aa_deg[:3] / max(symbols_to_rvdw(symbols))
        # / 0.02).astype(int) + 1)
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

        neigh_map = neighbour_matrix(_pp,
                                     symbols[_ind],
                                     cell_aa_deg=cell_aa_deg)

        # --- store batch info for assignment
        #     Caution: neigh_count may lead to (uncritical) overcounting if
        #              atom pairs are found in more than one batch

        neigh_count[_ind] += neigh_map.sum(axis=1)
        pair_list += [tuple(_ind2[_l]) for _l in np.argwhere(
                           neigh_map[noh[_ind]][:, noh[_ind]] == 1)]

    # --- This is a little fuzzy after various changes and methodology updates

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
                                         cell=cell_aa_deg), axis=-1)
        _d[_d == 0.0] = 'Inf'
        _i = np.argmin(_d)
        ass[_h] = ass[_i]

    # --- create ascending indices
    _n = 0
    _o = {}
    ass_n = []
    for _a in ass:
        if _a not in _o:
            _n += 1
            _o[_a] = _n
        ass_n.append(_o[_a])
    ass = np.array(ass_n)

    return ass - 1


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


def read_topology_file(fn):
    '''Returns dict of properties'''

    from ..read.coordinates import pdbReader, xyzReader
    from ..interface import cp2k

    fmt = fn.split('.')[-1]
    if fmt == 'pdb':
        # A little old messy code
        data, names, symbols, residues, cell_aa_deg, title = pdbReader(fn)
        # residues = np.array(residues).astype(str)
        resi = ['-'.join(_r) for _r in np.array(residues).astype(str)]
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
        if cell_aa_deg is None:
            cell_aa_deg = np.array(3*[0] + 3*[90.])

        return {'mol_map': mol_map,
                'symbols': symbols_to_symbols(symbols),  # redundant for PDB?
                'names': names,
                'residues': residues,
                'cell_aa_deg': cell_aa_deg,
                'fn_topo': fn}

    elif fmt == 'xyz':
        data, symbols, comments = xyzReader(fn)

        return {'symbols': symbols_to_symbols(symbols),
                'fn_topo': fn,
                }

    elif fmt in ['cp2k', 'restart', 'inp']:
        _C = cp2k.parse_restart_file(fn)
        syntax = [
            # attribute
            # location
            # keyword(s) (list=AND, tuple=OR)
            # process
            ('comments',
                ['GLOBAL'],
                'PROJECT_NAME',
                lambda x: x[0],
             ),
            ('cell_vec_aa',
                ['FORCE_EVAL', 'SUBSYS', 'CELL'],
                ['A', 'B', 'C'],
                lambda x: np.array(x).astype(float),
             ),
            ('abc',
                ['FORCE_EVAL', 'SUBSYS', 'CELL'],
                'ABC',
                lambda x: np.array(x).astype(float),
             ),
            ('albega',
                ['FORCE_EVAL', 'SUBSYS', 'CELL'],
                'ALBEGA',
                lambda x: np.array(x).astype(float),
             ),
            ('names',
                ['FORCE_EVAL', 'SUBSYS', 'COORD'],
                None,
                lambda x: x[0],
             ),
            ('pos_aa',
                ['FORCE_EVAL', 'SUBSYS', 'COORD'],
                None,
                lambda x: np.array(x[1:]).astype(float),
             ),
            ('vel_au',
                ['FORCE_EVAL', 'SUBSYS', 'VELOCITY'],
                None,
                lambda x: np.array(x).astype(float),
             ),
            ]
        _con = {}
        # --- read syntax
        for _attr in syntax:
            _attrname = _attr[0]
            _section = copy.deepcopy(_C)
            for _title in _attr[1]:
                _section = _section[_title]
            _section = _section['KEYWORDS']

            # --- what follows is walrus magic within a list comprehension :)
            _con[_attrname] = list(map(_attr[3], [
                _values[1:] if _keyword is not None
                else _values
                for _keyword in (
                    _attr[2] if isinstance(_attr[2], list)
                    else [_attr[2]]
                    )
                for _entry in _section
                if (_values := _entry.split())[0] == _keyword
                or _keyword is None
                ]))

        if (cell_aa_deg := _con['abc'] + _con['albega']) == []:
            cell_aa_deg = cell_l_deg(np.array(_con['cell_vec_aa']))

        if _con['vel_au'] == []:
            data = np.tile(np.concatenate(
                (_con['pos_aa'], np.zeros_like(_con['pos_aa'])),
                axis=-1
                ), (1, 1, 1))
        else:
            data = np.tile(np.concatenate(
                (_con['pos_aa'], _con['vel_au']),
                axis=-1
                ), (1, 1, 1))

        return {'symbols': symbols_to_symbols(tuple(_con['names'])),
                'names': tuple(_con['names']),
                'data_topo': data,
                'cell_aa_deg': cell_aa_deg,
                'comments_topo': _con['comments'],
                'fn_topo': fn}

    else:
        raise ValueError('Unknown format: %s.' % fmt)
