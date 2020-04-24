#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


import numpy as np
import copy
import warnings as _warnings

from ..physics import constants
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..mathematics.algebra import kabsch_algorithm, rotate_vector

# NB: the molecules have to be sequentially numbered starting with 0
# the script will transform them starting with 0
# Get rid of class dependence


def dist_crit_aa(symbols):
    '''Get distance criteria matrix of symbols (in angstrom)
       http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html
       '''
    natoms = len(symbols)
    crit_aa = np.zeros((natoms, natoms))
    _r = np.array(constants.symbols_to_rvdw(symbols)) / 100.0
    crit_aa = (_r[:, None] + _r[None, :])
    crit_aa *= 0.6
    return crit_aa


def dec(prop, indices, n_ind=None):
    """decompose prop according to indices
       n_ind: if number of pieces cannot be retrieved from indices"""
    if n_ind is None:
        n_ind = max(indices)+1
    return [
        np.array([
            prop[k] for k, j_mol in enumerate(indices) if j_mol == i_mol
            ]) for i_mol in range(n_ind)
        ]


def cowt(pos, wt, **kwargs):
    '''Calculate centre of weight, consider periodic boundaries before
       calling this method.'''

    _wt = np.array(wt)
    _axis = kwargs.get("axis", -2)
    _sub = kwargs.get('subset', slice(None))
    _p = np.moveaxis(pos, _axis, 0)
    _slc = (_sub,) + (len(_p.shape)-1) * (None,)

    return np.sum(_p[_sub] * _wt[_slc], axis=0) / _wt[_sub].sum()


def get_cell_vec(cell, n_fields=3, priority=(0, 1, 2)):
    '''cell as np.array/list of: a b c al be ga
       n_fields: usually 3

       Priority defines the alignment of non-rectangular objects in cartesian
       space; by convention the z axis is the odd one (i.e. not aligned), and
       cell vectors are calculated accordingly (e.g. in VMD); here, this
       corresponds to priority (0,1,2). Any deviation (e.g. z-axis IS aligned,
       alpha > 90°, ...) can be taken account for by adjusting priority, and
       the correct cell vectors are calculated. However, VMD or other
       programmes may still calculate wrong cell vectors. Hence, ...
       Priority should always be (0,1,2) and symmetry conventions be used
       (e.g. for monoclinic cells: beta is the angle > 90°; CPMD wants alpha
       to be >90° but this is wrong and CELL VECTORS should be used instead)
       '''

    abc, albega = cell[:3], cell[3:] * np.pi / 180.
    cell_vec = np.zeros((3, n_fields))
    v0, v1, v2 = priority
    cell_vec[v0, v0] = abc[v0]
    cell_vec[v1, v1] = abc[v1] * np.sin(albega[(3 - v0 - v1)])
    cell_vec[v1, v0] = abc[v1] * np.cos(albega[(3 - v0 - v1)])
    cell_vec[v2, v2] = abc[v2] * np.sin(albega[(3 - v0 - v2)]) \
        * np.sin(albega[(3 - v1 - v2)])
    cell_vec[v2, v0] = abc[v2] * np.cos(albega[(3 - v0 - v2)])
    cell_vec[v2, v1] = abc[v2] * np.cos(albega[(3 - v1 - v2)])

    return cell_vec


def detect_lattice(cell_aa_deg, priority=(0, 1, 2)):
    '''Obtain lattice system from cell measures.
       Does not care of axis order priority.
       (Beta)'''
    if cell_aa_deg is None or np.any(cell_aa_deg == 0.):
        _warnings.warn("Got empty cell!", RuntimeWarning, stacklevel=2)
        return None

    abc, albega = cell_aa_deg[:3], cell_aa_deg[3:]
    _a = np.invert(np.diff(abc).astype(bool))
    _b = np.invert(np.diff(albega).astype(bool))

    if np.all(albega == 90.0):
        if np.all(_a):
            return 'cubic'
        elif not np.any(_a):
            return 'orthorhombic'
        else:
            return 'tetragonal'

    if np.sum(albega == 90.0) == 2:
        if 120 in albega and np.any(_a * _b):
            return 'hexagonal'
        elif not np.any(_a):
            return 'monoclinic'
        else:
            _warnings.warn("Unusual lattice!", RuntimeWarning, stacklevel=2)
            return 'triclinic'

    elif np.all(abc) and np.all(albega):
        return 'rhombohedral'

    else:
        return 'triclinic'


def wrap(pos_aa, cell_aa_deg, **kwargs):
    '''pos_aa: shape ([n_frames,] n_atoms, three)
       cell: [ a b c al be ga ]'''

    lattice = detect_lattice(cell_aa_deg)
    if lattice is not None:
        # python3.8: use walrus
        if lattice in ['cubic', 'orthorhombic', 'tetragonal']:
            # --- fast
            return pos_aa - np.floor(pos_aa/cell_aa_deg[:3]) * cell_aa_deg[:3]

        else:
            # --- more expensive (ToDo: optimise tensordot, ceb; has np.cross)
            cell_vec_aa = get_cell_vec(cell_aa_deg)  # checked: inexpensive
            return pos_aa - np.tensordot(np.floor(ceb(pos_aa, cell_vec_aa)),
                                         cell_vec_aa,
                                         axes=1
                                         )
    else:
        return pos_aa


def distance_pbc(p0, p1, cell=None, **kwargs):
    '''p1 – p0 with or without periodic boundaries
       accepts cell_aa_deg argument
       length units need not be in angstrom, but
       have to be consistent between p0, p1, and
       cell.
       '''
    # actually it does not calculate a "distance"
    _d = p1 - p0
    if cell is not None:
        _d2 = _d - _pbc_shift(_d, cell)
        # _d3 = min(list(_d.flatten()),
        #           list(_d2.flatten()),
        #           key=abs)
        # _d = np.array(_d3).reshape(_d.shape)
        _d = _d2
    return _d


def _pbc_shift(_d, cell):
    '''_d in aa of shape ...
       cell: [ a b c al be ga ]'''

    if not any([_a <= 0.0 for _a in cell[:3]]):
        if not all([_a == 90.0 for _a in cell[3:]]):
            cell_vec = get_cell_vec(cell)
            _c = ceb(_d, cell_vec)
            return np.tensordot(np.around(_c), cell_vec, axes=1)
        else:
            return np.around(_d/cell[:3]) * cell[:3]
    else:
        return np.zeros_like(_d)


def distance_matrix(p0, p1=None, cell=None, cartesian=None):
    '''Expects one or two args of shape (n_atoms, three) ... (FRAME).
       Order: p0, p1 ==> d = p1 - p0

       Supports periodic boundaries (give cell as [x, y, z, al, be, ga];
                                     angles in degrees).
       '''
    # ToDo: the following lines explode memory for many atoms
    #   ==> do coarse mapping beforehand
    # (overlapping batches) or set a max limit for n_atoms
    if p1 is None:
        p1 = p0

    if max(p0.shape[0], p1.shape[0]) > 1000:
        # python3.8: use walrus
        print(max(p0.shape[0], p1.shape[0]))
        raise MemoryError('Too many atoms for molecular recognition'
                          '(>1000 atom support in a future version)!'
                          )
    dist_array = distance_pbc(p0[:, None], p1[None, :], cell=cell)

    if cartesian is not None:
        return dist_array
    else:
        return np.linalg.norm(dist_array, axis=-1)


def neighbour_matrix(pos_aa, symbols, **kwargs):
    '''Create sparse matrix with entries 1 for neighbouring atoms.
       Expects positions in angstrom of shape (n_atoms, three).
       '''
    symbols = np.array(symbols)
    cell_aa_deg = kwargs.get("cell_aa_deg")
    dist_array = distance_matrix(pos_aa, cell=cell_aa_deg)
    dist_array[dist_array == 0.0] = 'Inf'
    crit_aa = dist_crit_aa(symbols)

    return dist_array <= crit_aa


def join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs):
    '''pos_aa (in angstrom) with shape ([n_frames,] n_atoms, three)
    Has still problems with cell-spanning molecules
    Molecules have to be numbered starting with 0!'''
    if 0 not in mol_map:
        raise TypeError('Given mol_map not an enumeration of indices!' %
                        mol_map)
    pos_aa = np.moveaxis(pos_aa, -2, 0)
    _shape = pos_aa.shape
    n_atoms = _shape[0]
    w = kwargs.get('weights', np.ones((n_atoms)))
    w = dec(w, mol_map)

    _pos_aa = dec(pos_aa, mol_map)
    mol_com_aa = []
    for _i, (_w, _p) in enumerate(zip(w, _pos_aa)):
        # actually: needs connectivity pattern to wrap everything correctly
        # the slightly expensive matrix analysis improves the result

        # --- ToDo: awkward check if _p has frames (frame 0 as reference)
        if len(_p.shape) == 3:
            _p_ref = _p[:, 0]
        else:
            _p_ref = _p
        # --- find atom that is closest to its counterparts
        # --- ToDo: NOT WORKING well for cell-spanning molecules
        _r = np.argmin(np.linalg.norm(distance_matrix(
                                                      _p_ref,
                                                      cell=cell_aa_deg,
                                                      ),
                                      axis=1))
        # --- alternative: use the heaviest atom as reference
        # _r = np.argmax(_w)

        # --- complete mols
        _p -= _pbc_shift(_p - _p[_r, :], cell_aa_deg)
        c_aa = cowt(_p, _w, axis=0)
        mol_com_aa.append(c_aa)
        _p -= c_aa[None, :]

    # --- wrap = performance bottleneck? --> definitely for non-tetragonal lat
    mol_com_aa = wrap(np.array(mol_com_aa), cell_aa_deg)

    # --- wrap set of pos
    for _i, _com in enumerate(mol_com_aa):
        # --- in-loop to retain order (np.argsort=unreliable)
        ind = np.array(mol_map) == _i
        pos_aa[ind] = _pos_aa[_i] + _com[None, :]

    return np.moveaxis(pos_aa, 0, -2), np.moveaxis(np.array(mol_com_aa), 0, -2)


def get_atom_spread(pos):
    return np.array([np.amax(_p) - np.amin(_p)
                     for _p in np.moveaxis(pos, -1, 0)])


def align_atoms(pos_mobile, w, **kwargs):
    '''Align atoms within trajectory or towards an external
       reference. Kinds and order of atoms (usually) have to
       be equal.
       Specify additional atom data (e.g., velocities),
       which has to be parallel transformed, listed through the keyword
       data=... (shape has to be according to positions).
       '''

    w = np.array(w)
    _sub = kwargs.get('subset', slice(None))
    _data = kwargs.get('data')

    pos_mob = copy.deepcopy(pos_mobile)
    # --- get subset data sets
    # --- default reference: frame 0
    _s_pos_ref = copy.deepcopy(kwargs.get('ref', pos_mobile[0])[_sub])
    _s_pos_mob = pos_mob[:, _sub]

    # --- get com of data sets
    com_ref = cowt(_s_pos_ref, w[_sub], axis=-2)
    com_mob = cowt(_s_pos_mob, w[_sub], axis=-2)

    # --- apply com (reference can be frame or trajectory)
    _i_s_pos_ref = np.moveaxis(_s_pos_ref, -2, 0)
    _s_pos_ref = np.moveaxis(_i_s_pos_ref - com_ref[(None,)], 0, -2)
    _i_pos_mob = np.moveaxis(pos_mob, -2, 0)
    pos_mob = np.moveaxis(_i_pos_mob - com_mob[(None,)], 0, -2)
    del _i_s_pos_ref, _i_pos_mob

    for frame, P in enumerate(_s_pos_mob):
        U = kabsch_algorithm(P * w[_sub, None], _s_pos_ref * w[_sub, None])
        pos_mob[frame] = rotate_vector(pos_mob[frame], U)
        if _data is not None:
            for _d in _data:
                _d[frame] = rotate_vector(_d[frame], U)

    # --- define return shift
    com_return = com_ref

    _slc = (len(pos_mob.shape) - len(com_return.shape)) * (None,)
    pos_mob += com_return[_slc]

    if _data is not None:
        return pos_mob, _data
    else:
        return pos_mob


def find_methyl_groups(pos, symbols, hetatm=False, **kwargs):
    '''pos of shape (n_atoms, n_fields) (FRAME)
       Outformat is C H H H'''

    dist_array = distance_matrix(pos, cell=kwargs.get("cell_aa_deg"))
    n_atoms = len(symbols)
    symbols = np.array(symbols)

    indh = symbols == 'H'
    if hetatm:
        ind = symbols != 'H'
    else:
        ind = symbols == 'C'
    ids = np.arange(n_atoms) + 1
    out = [[ids[i], [ids[j] for j in range(n_atoms)
            if dist_array[i, j] < dist_crit_aa(symbols)[i, j] and indh[j]]]
           for i in range(n_atoms) if ind[i]]

    out = '\n'.join(['A%03d ' % i[0]+'A%03d A%03d A%03d' % tuple(i[1])
                     for i in out if len(i[1]) == 3])

    print(out)
