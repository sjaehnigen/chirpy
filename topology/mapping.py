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
import sys
import copy

from ..physics import constants

# NB: the molecules have to be sequentially numbered starting with 0
# the script will transform them starting with 0


def dist_crit_aa(symbols):
    '''Get distance criteria matrix of symbols (in angstrom)'''
    natoms = len(symbols)
    crit_aa = np.zeros((natoms, natoms))
    _r = np.array(constants.symbols_to_rvdw(symbols)) / 100.0
    crit_aa = (_r[:, None] + _r[None, :])
    crit_aa *= 0.6  # http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.1/ug/node26.html
    return crit_aa


def dec(prop, indices):
    """decompose prop according to indices"""
    return [
        np.array([
            prop[k] for k, j_mol in enumerate(indices) if j_mol == i_mol
            ]) for i_mol in range(max(indices)+1)
        ]


def get_atom_spread(pos):
    return np.array([np.amax(_p) - np.amin(_p) for _p in np.moveaxis(pos, -1, 0)])


def map_atoms_by_coordinates(mol1, mol2):
    '''expects two Molecule objects of the same shape. I use frame 0 of mol2 as reference. Not tested for long trajectories of mol1.'''
# pos_aa,sym,box_aa,replica,vec_trans,pos_cryst,sym_cr):
    d1 = mol1.XYZData
    d2 = mol2.XYZData
    ie, tmp = d1._is_similar(d2)
    if not ie:
        print('ERROR: The two Molecule objects are not similar!\n n_atoms: %s\n n_fields: %s\n symbols: %s'%tuple(tmp))
        sys.exit(1)
#    if hasattr(mol1,'UnitCell'):
#        cell_aa = mol1.UnitCell.abc #only albega = 90 degreee for now

    def get_assign(pos1, pos2, unit_cell=None):
        dist_array = pos1[:, :, None, :] - pos2[0, None, None, :, :]
        if unit_cell:
            dist_array -= np.around(dist_array/unit_cell.abc)*unit_cell.abc
        return np.argmin(np.linalg.norm(dist_array, axis=-1), axis=2)
        # insert routine to avoid double index

    assign = np.zeros((d1.n_frames, d1.n_atoms)).astype(int)
    for s in np.unique(d1.symbols):
        i1 = d1.symbols == s
        i2 = d2.symbols == s
        ass = get_assign(d1.data[:, i1, :3], d2.data[:, i2, :3], unit_cell=getattr(mol1, 'UnitCell', None))
        assign[:, i1] = np.array([np.arange(d2.n_atoms)[i2][ass[fr]] for fr in range(d1.n_frames)]) #maybe unfortunate to long trajectories
    return assign


def kabsch_algorithm(P, ref):
    C = np.dot(np.transpose(ref), P)
    V, S, W = np.linalg.svd(C)

    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    # Create Rotation matrix U
    return np.dot(V, W)


def align_atoms(pos_mobile, w, **kwargs):
    '''Demands a reference for each frame, respectively'''
    _sub = kwargs.get('subset', slice(None))
    pos_mob = copy.deepcopy(pos_mobile)
    _s_pos_ref = kwargs.get('ref', pos_mobile[0])[_sub]
    _s_pos_mob = pos_mob[:, _sub]
    _s_pos_ref -= (np.sum(_s_pos_ref * w[_sub, None],
                   axis=-2)/w[_sub].sum())[None, :]
    com = np.sum(_s_pos_mob*w[None, _sub, None], axis=-2)/w[_sub].sum()
    pos_mob -= com[:, None, :]
    for frame, P in enumerate(_s_pos_mob):
        U = kabsch_algorithm(P*w[_sub, None], _s_pos_ref*w[_sub, None])
        pos_mob[frame] = np.tensordot(U, pos_mob[frame],
                                      axes=([1], [1])).swapaxes(0, 1)
    pos_mob += com[:, None, :]
    return pos_mob
