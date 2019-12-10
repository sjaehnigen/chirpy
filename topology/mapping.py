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
import copy
import warnings as _warnings

from ..physics import constants
from ..mathematics.algebra import change_euclidean_basis as ceb
from ..mathematics.algebra import kabsch_algorithm

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


def dec(prop, indices):
    """decompose prop according to indices"""
    return [
        np.array([
            prop[k] for k, j_mol in enumerate(indices) if j_mol == i_mol
            ]) for i_mol in range(max(indices)+1)
        ]


def get_cell_vec(cell_aa_deg, n_fields=3, priority=(0, 1, 2)):
    '''cell_aa_deg as np.array/list of: a b c al be ga
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

    abc, albega = cell_aa_deg[:3], cell_aa_deg[3:] * np.pi / 180.
    cell_vec_aa = np.zeros((3, n_fields))
    v0, v1, v2 = priority
    cell_vec_aa[v0, v0] = abc[v0]
    cell_vec_aa[v1, v1] = abc[v1] * np.sin(albega[(3 - v0 - v1)])
    cell_vec_aa[v1, v0] = abc[v1] * np.cos(albega[(3 - v0 - v1)])
    cell_vec_aa[v2, v2] = abc[v2] * np.sin(albega[(3 - v0 - v2)]) \
        * np.sin(albega[(3 - v1 - v2)])
    cell_vec_aa[v2, v0] = abc[v2] * np.cos(albega[(3 - v0 - v2)])
    cell_vec_aa[v2, v1] = abc[v2] * np.cos(albega[(3 - v1 - v2)])

    return cell_vec_aa


def detect_lattice(cell_aa_deg, priority=(0, 1, 2)):
    '''Obtain lattice system from cell measures.
       Does not care of axis order priority.
       (Beta)'''
    if cell_aa_deg is None or np.any(cell_aa_deg == 0.):
        _warnings.warn("Got empty cell. Cannot detect symmetry!",
                       UserWarning)
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
            _warnings.warn("Strange symmetry found!", UserWarning)
            return 'triclinic'

    elif np.all(abc) and np.all(albega):
        return 'rhombohedral'

    else:
        return 'triclinic'


def wrap(pos_aa, cell_aa_deg, **kwargs):
    '''pos_aa: shape (n_frames, n_atoms, three) or (n_atoms, three)
       cell: [ a b c al be ga ]'''
    cell_vec_aa = get_cell_vec(cell_aa_deg)

    if not any([_a <= 0.0 for _a in cell_aa_deg[:3]]):
        return pos_aa - np.tensordot(np.floor(ceb(pos_aa, cell_vec_aa)),
                                     cell_vec_aa,
                                     axes=1
                                     )
    else:
        _warnings.warn('Cell size zero!', UserWarning)
        return pos_aa


def distance_pbc(p1, p0, **kwargs):
    # actually it does not calculate a "distance"
    _d = p1 - p0
    try:
        return _d - _pbc_shift(_d, kwargs.get("cell_aa_deg"))
    except TypeError:
        return _d


def _pbc_shift(_d, cell_aa_deg):
    '''_d in aa of shape ...'''
    if not any([_a <= 0.0 for _a in cell_aa_deg[:3]]):
        if not all([_a == 90.0 for _a in cell_aa_deg[3:]]):
            cell_vec_aa = get_cell_vec(cell_aa_deg)
            _c = ceb(_d, cell_vec_aa)
            return np.tensordot(np.around(_c), cell_vec_aa, axes=1)
        else:
            return np.around(_d/cell_aa_deg[:3]) * cell_aa_deg[:3]
    else:
        return np.zeros_like(_d)


def get_distance_matrix(*args, **kwargs):
    '''one or two args of shape (n_atoms, three) ... (FRAME)'''
    # ToDo: the following lines explode memory for many atoms
    #   ==> do coarse mapping beforehand
    # (overlapping batches) or set a max limit for n_atoms
    if len(args) == 1:
        _p0 = _p1 = args[0]
    elif len(args) == 2:
        _p0, _p1 = args
    else:
        raise TypeError('More than two arguments given!')

    if max(_p0.shape[0], _p1.shape[0]) > 1000:
        print(max(_p0.shape[0], _p1.shape[0]))
        raise MemoryError('Too many atoms for molecular recognition'
                          '(>1000 atom support in a future version)!'
                          )
    dist_array = distance_pbc(_p1[:, None, :], _p0[None, :, :], **kwargs)
    # dist_array = pos_aa[:, None, :] - pos_aa[None, :, :]
    # cell_aa_deg = kwargs.get("cell_aa_deg")
    # if cell_aa_deg is not None:
    #    dist_array -= _pbc_shift(dist_array, cell_aa_deg)

    if kwargs.get("cartesian") is not None:
        return dist_array
    else:
        return np.linalg.norm(dist_array, axis=-1)


def cowt(pos_aa, wt, **kwargs):
    '''Calculate centre of weight, consider periodic boundaries before
       calling this method.'''

    _wt = np.array(wt)
    _axis = kwargs.get("axis", 1)
    _sub = kwargs.get('subset', slice(None))
    _p = np.moveaxis(pos_aa, _axis, 0)
    _slc = (_sub,) + (len(_p.shape)-1) * (None,)

    return np.sum(_p[_sub] * _wt[_slc], axis=0) / _wt[_sub].sum()


def wrap_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs):
    '''DEPRECATED, use join_molecules()'''
    join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs)


def join_molecules(pos_aa, mol_map, cell_aa_deg, **kwargs):
    '''pos_aa (in angstrom) with shape ( n_frames, n_atoms, three )
    Has still problems with cell-spanning molecules'''
    n_frames, n_atoms, three = pos_aa.shape
    w = kwargs.get('weights', np.ones((n_atoms)))
    w = dec(w, mol_map)

    _pos_aa = copy.deepcopy(pos_aa)
    mol_c_aa = []
    for i_mol in set(mol_map):
        ind = np.array(mol_map) == i_mol
        _p = _pos_aa[:, ind]
        # reference atom: 0 (this may lead to problems, say, for large,
        # cell-spanning molecules)
        # _r = [0]*sum(ind)
        # choose as reference atom the one with smallest distances to all atoms
        # (works better but still not perfect; needs an adaptive scheme)
        # actually: needs connectivity pattern to wrap everything correctly

        _r = np.argmin(np.linalg.norm(get_distance_matrix(
                                                    _p[0],
                                                    cell_aa_deg=cell_aa_deg,
                                                    ),
                                      axis=1))
        _r = [_r]*sum(ind)
        _p -= _pbc_shift(_p - _p[:, _r, :], cell_aa_deg)
        c_aa = cowt(_p, w[i_mol])
        mol_c_aa.append(wrap(c_aa, cell_aa_deg))
        _pos_aa[:, ind] = _p - (c_aa - mol_c_aa[-1])[:, None, :]

    return _pos_aa, mol_c_aa


def get_atom_spread(pos):
    return np.array([np.amax(_p) - np.amin(_p)
                     for _p in np.moveaxis(pos, -1, 0)])


def align_atoms(pos_mobile, w, **kwargs):
    '''Demands a reference for each frame, respectively'''

    _sub = kwargs.get('subset', slice(None))

    pos_mob = copy.deepcopy(pos_mobile)
    _s_pos_ref = copy.deepcopy(kwargs.get('ref', pos_mobile[0])[_sub])
    _s_pos_mob = pos_mob[:, _sub]

    com_ref = cowt(_s_pos_ref, w[_sub], axis=-2)
    com_mob = cowt(_s_pos_mob, w[_sub], axis=-2)

    # DEVEL: reference can be frame or trajectory
    _i_s_pos_ref = np.moveaxis(_s_pos_ref, -2, 0)
    _s_pos_ref = np.moveaxis(_i_s_pos_ref - com_ref[(None,)], 0, -2)

    _i_pos_mob = np.moveaxis(pos_mob, -2, 0)
    pos_mob = np.moveaxis(_i_pos_mob - com_mob[(None,)], 0, -2)
    # pos_mob -= com_mob[:, None, :]

    for frame, P in enumerate(_s_pos_mob):
        U = kabsch_algorithm(P*w[_sub, None], _s_pos_ref*w[_sub, None])
        pos_mob[frame] = np.tensordot(U, pos_mob[frame],
                                      axes=([1], [1])).swapaxes(0, 1)

    com_return = com_ref
    _slc = (len(pos_mob.shape) - len(com_return.shape)) * (None,)
    pos_mob += com_return[_slc]

    return pos_mob


def find_methyl_groups(pos, symbols, hetatm=False, **kwargs):
    '''pos of shape (n_atoms, n_fields) (FRAME)
       Outformat is C H H H'''
    dist_array = get_distance_matrix(pos, **kwargs)
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
