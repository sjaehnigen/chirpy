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

import copy
import warnings
import numpy as np
from functools import partial

from ..topology.dissection import fermi_cutoff_function
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
        _trans = positions - _o[:, None, :]
        if cell is not None:
            _trans -= np.around(_trans/cell) * cell
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


# ToDO: tobedel
def get_vibrational_spectrum_b(cur_dipoles, mag_dipoles, positions, **kwargs):
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
    subparticle = kwargs.get('subparticle')
    # BETA: correlate moment of one part with another part,
    #     expects tuple of integers
    #     notparticles: exclude these indices; expects list of integers
    subparticles = kwargs.get('subparticles')
    subnotparticles = kwargs.get('subnotparticles')
    # BETA: remove direct correlation between background
    cutoff_bg = kwargs.get('background_correction_cutoff')

    spectrum = list()
    for _i, _o in enumerate(origins):
        _c = copy.deepcopy(cur_dipoles)
        _m = copy.deepcopy(mag_dipoles)

        # R_I(t) - R_J(t) #used in manuscript
        # USE ChirPy for this
        _trans = positions - _o[:, None, :]
        if cell is not None:
            _trans -= np.around(_trans/cell) * cell

        # R_I(t) - R_J(0) #gauge invariant according to Rodolphe/Arne
        # _trans = pos_au-_o[0,None,None,:]
        # _trans -= np.around(_trans/cell_au)*cell_au
        # _trans += _o[:,None,:] - _o[0,None,None,:]
        _m += magnetic_dipole_shift_origin(_c, _trans)

        if cut_type == 'soft':  # for larger cutoffs
            _scal = fermi_cutoff_function(
                            np.linalg.norm(_trans, axis=2),
                            cutoff,
                            0.125 * constants.l_aa2au
                            )
            warnings.warn("Still using angstrom here!")
            _c *= _scal[:, :, None]
            _m *= _scal[:, :, None]

        if cut_type == 'hard':  # cutoff <2 aa
            _ind = np.linalg.norm(_trans, axis=2) > cutoff
            _c[_ind, :] = np.array([0.0, 0.0, 0.0])
            _m[_ind, :] = np.array([0.0, 0.0, 0.0])

        if type(cutoff_bg) is float:
            _c_bg = copy.deepcopy(_c)
            # only direct correlation, no transport term!
            # _m_bg = copy.deepcopy(m)
            # --- complete background
            _m_bg = copy.deepcopy(_m)
            if cut_type == 'soft':
                _m_bg *= _scal[:, :, None]
            if cut_type == 'hard':
                _m[_ind, :] = np.array([0.0, 0.0, 0.0])
            _ind_bg = np.linalg.norm(
                   _trans, axis=2) <= cutoff_bg
            _c_bg[_ind_bg, :] = np.array([0.0, 0.0, 0.0])
            _m_bg[_ind_bg, :] = np.array([0.0, 0.0, 0.0])

        if all([_k is None for _k in [
                                subparticle,
                                subparticles,
                                subnotparticles
                                ]]):
            _c = _c.sum(axis=1)
            _m = _m.sum(axis=1)
            omega, ira, C_ira = spectral_density(_c, **kwargs)
            omega, vcd, C_vcd = spectral_density(_c, _m, **kwargs)
            if type(cutoff_bg) is float:
                _c_bg = _c_bg.sum(axis=1)
                _m_bg = _m_bg.sum(axis=1)
                ira -= spectral_density(_c_bg, **kwargs)[1]
                vcd -= spectral_density(_c_bg, _m_bg, **kwargs)[1]

        elif type(subparticle) is int:
            _c1 = _c.sum(axis=1)
            _c2 = _c[:, subparticle]
            _m1 = _m.sum(axis=1)
            _m2 = _m[:, subparticle]
            omega, ira, C_ira = spectral_density(_c1, _c2, **kwargs)
            omega, vcd1, C_vcd1 = spectral_density(_c1, _m2, **kwargs)
            omega, vcd2, C_vcd2 = spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1 + vcd2) / 2

            if type(cutoff_bg) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = _c_bg[:, subparticle]
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = _m_bg[:, subparticle]
                ira -= spectral_density(_c1_bg, _c2_bg, **kwargs)[1]
                vcd -= 0.5 * spectral_density(_c1_bg, _m2_bg, **kwargs)[1]
                vcd -= 0.5 * spectral_density(_c2_bg, _m1_bg, **kwargs)[1]

        elif type(subnotparticles) is list:
            _c1 = _c.sum(axis=1)
            _c2 = copy.deepcopy(_c)
            _c2[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
            _c2 = _c2.sum(axis=1)
            _m1 = _m.sum(axis=1)
            _m2 = copy.deepcopy(_m)
            _m2[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
            _m2 = _m2.sum(axis=1)
            omega, ira, C_ira = spectral_density(_c1, _c2, **kwargs)
            omega, vcd1, C_vcd1 = spectral_density(_c1, _m2, **kwargs)
            omega, vcd2, C_vcd2 = spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1 + vcd2) / 2
            if type(cutoff_bg) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = copy.deepcopy(_c_bg)
                _c2_bg[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
                _c2_bg = _c2_bg.sum(axis=1)
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = copy.deepcopy(_m_bg)
                _m2_bg[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
                _m2_bg = _m2_bg.sum(axis=1)
                ira -= spectral_density(_c1_bg, _c2_bg, **kwargs)[1]
                vcd -= 0.5 * spectral_density(_c1_bg, _m2_bg, **kwargs)[1]
                vcd -= 0.5 * spectral_density(_c2_bg, _m1_bg, **kwargs)[1]

        elif type(subparticles) is tuple:
            _c1 = _c[:, subparticles[0]]
            _c2 = _c[:, subparticles[1]]
            _m1 = _m[:, subparticles[0]]
            _m2 = _m[:, subparticles[1]]
            omega, ira, C_ira = spectral_density(_c1, _c2, **kwargs)
            omega, vcd1, C_vcd1 = spectral_density(_c1, _m2, **kwargs)
            omega, vcd2, C_vcd2 = spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1 + vcd2) / 2
            if type(cutoff_bg) is float:
                raise NotImplementedError(
                                    'Background correction not '
                                    'implemented for subparticles option!'
                                    )

        else:
            raise ValueError('Subparticle(s) is (are) not an integer '
                             '(a tuple of integers)!')

        # sort this out!
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
