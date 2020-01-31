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

import copy
import warnings
import numpy as np
from functools import partial

from ..topology.dissection import fermi_cutoff_function
from ..topology.mapping import distance_pbc
from .classical_electrodynamics import magnetic_dipole_shift_origin
from .statistical_mechanics import spectral_density
from ..physics import constants


def _apply_cutoff(_x, _pos, _cutoff, _cut_type):
    _d = np.linalg.norm(_pos, axis=2)

    if _cut_type == 'soft':  # for larger cutoffs
        _D = 0.125 * constants.l_aa2au
        warnings.warn("Still using angstrom here!", stacklevel=2)
        _scal = fermi_cutoff_function(_d, _cutoff, _D)
        _x *= _scal[:, :, None]

    elif _cut_type == 'hard':  # cutoff <2 aa
        _ind = _d > _cutoff
        _x[_ind, :] = np.array([0.0, 0.0, 0.0])

    return _x


def get_power_spectrum(velocities, **kwargs):
    '''Expects data of shape (n_frames, n_atoms, three)
       No support of trajectory iterators.
       The output is averaged over the no. of atoms/species.
       Optionally returns time-correlation function (return_tcf=True).
       '''
    n_frames, n_atoms, three = velocities.shape
    wgh = kwargs.get('weights', np.ones(n_atoms))
    r_tcf = kwargs.get('return_tcf', False)
    wgh = np.array(wgh)
    omega, S, R = zip(*[spectral_density(_v, **kwargs)
                        for _v in velocities.swapaxes(0, 1)])

    data = {}
    data['power'] = (np.array(S) * wgh[:, None]).sum(axis=0) / n_atoms
    data['omega'] = omega[0]
    if r_tcf:
        data['tcf_velocities'] = np.array(R).sum(axis=0) / n_atoms

    return data


def get_vibrational_spectrum(cur_dipoles, mag_dipoles, positions, **kwargs):
    '''Expects dipole data of shape (n_frames, n_kinds, three)
       No support of trajectory iterators.
       The output is averaged over the no. of origins
       Expects atomic units (but it is not mandatory).
       Optionally returns time-correlation function (return_tcf=True).
       '''
    origins = kwargs.get('origins', np.zeros((1, 1, 3)))
    cell = kwargs.get('cell')
    cutoff = kwargs.get('cutoff', 0)
    cut_type = kwargs.get('cut_type', 'soft')

    r_tcf = kwargs.get('return_tcf', False)

    # BETA: correlate moment of part with total moments, expects one integer
    sub = kwargs.get('subparticle')
    # BETA: correlate moment of one part with another part,
    #     expects tuple of integers
    #     notparticles: exclude these indices; expects list of integers
    subs = kwargs.get('subparticles')
    subnots = kwargs.get('subnotparticles')
    # BETA: remove direct correlation between background
    cutoff_bg = kwargs.get('background_correction_cutoff')

    spectrum = list()
    for _i, _o in enumerate(origins):
        # print(_i)
        _c = copy.deepcopy(cur_dipoles)
        _m = copy.deepcopy(mag_dipoles)

        # --- gauge-transport

        _trans = distance_pbc(_o[:, None], positions, cell_aa_deg=cell)
        _m += magnetic_dipole_shift_origin(_c, _trans)

        # --- apply the given cutoff either the hard or soft way
        _c = _apply_cutoff(_c, _trans, cutoff, cut_type)
        _m = _apply_cutoff(_m, _trans, cutoff, cut_type)

        # --- define background
        if cutoff_bg is not None:
            _c_bg = copy.deepcopy(_c)
            _m_bg = copy.deepcopy(_m)
            _c_bg = _apply_cutoff(_c_bg, _trans, cutoff_bg, 'hard')
            _m_bg = _apply_cutoff(_m_bg, _trans, cutoff_bg, 'hard')

        # --- some definitions of aux functions
        def _get_spectra(_a, _b, **kwargs):
            _a = _a.sum(axis=1)
            _b = _b.sum(axis=1)

            omega, S_aa, R_aa = spectral_density(_a, **kwargs)
            omega, S_ab, R_ab = spectral_density(_a, _b, **kwargs)

            return omega, S_aa, S_ab, R_aa, R_ab

        def _get_subspectra(_a, _b, _sub1, _sub2=None, **kwargs):
            _a1 = _a[:, _sub1]
            _b1 = _b[:, _sub1]
            if _sub2 is None:
                _a2 = _a.sum(axis=1)
                _b2 = _b.sum(axis=1)
            else:
                _a2 = _a[:, _sub2]
                _b2 = _b[:, _sub2]

            omega, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            omega, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            omega, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)

            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2

            return omega, S_aa, S_ab, R_aa, R_ab

        def _get_subnotspectra(_a, _b, _subnot, **kwargs):
            _a1 = _a.sum(axis=1)
            _b1 = _b.sum(axis=1)
            _a2 = copy.deepcopy(_a)
            _b2 = copy.deepcopy(_b)
            _a2[:, _subnot] = np.array([0.0, 0.0, 0.0])
            _b2[:, _subnot] = np.array([0.0, 0.0, 0.0])
            _a2 = _a2.sum(axis=1)
            _b2 = _b2.sum(axis=1)

            omega, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            omega, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            omega, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)

            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2

            return omega, S_aa, S_ab, R_aa, R_ab

        if sub is not None:
            if not isinstance(sub, int):
                raise TypeError('Subparticle has to be an integer!')
            _get = partial(_get_subspectra, _sub=sub)

        elif subnots is not None:
            if not isinstance(subnots, list):
                raise TypeError('Subnotparticles has to be a list!')
            _get = partial(_get_subnotspectra, _subnot=subnots)

        elif subs is not None:
            if not isinstance(subs, tuple):
                raise TypeError('Subparticles has to be a tuple of integers!')
            _get = partial(_get_subspectra, _sub=subs[0], _sub2=subs[1])

        else:
            _get = _get_spectra

        # --- get spectra
        omega, ira, vcd, C_ira, C_vcd = _get(_c, _m, **kwargs)

        if cutoff_bg is not None:
            _tmp = _get(_c_bg, _m_bg, **kwargs)
            ira -= _tmp[1]
            vcd -= _tmp[2]
            C_ira -= _tmp[3]
            C_vcd -= _tmp[4]

        # ToDo: sort this out!
        # _cc = constants.current_current_prefactor(300)
        # _cm = constants.current_magnetic_prefactor(omega, 300)
        _cc = 1.
        _cm = 4. * omega
        # --- prefactor for C functions?
        spectrum.append([omega, ira * _cc, vcd * _cm, C_ira, C_vcd])

    # --- average over origins
    spectrum = np.array(spectrum).sum(axis=0) / origins.shape[0]

    data = {}
    data['omega'] = spectrum[0]
    data['va'] = spectrum[1]
    data['vcd'] = spectrum[2]

    if r_tcf:
        warnings.warn("Return tcf incomplete!", stacklevel=2)
        data['tcf_va'] = spectrum[3]
        data['tcf_vcd'] = spectrum[4]
    return data
