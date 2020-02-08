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


def power_from_tcf(velocities, **kwargs):
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


def absorption_from_transition_moments(etdm_au):
    '''absorption in km/mol'''
    return (etdm_au ** 2).sum(axis=-1) * constants.IR_au2kmpmol


def _get_tcf_spectrum(*args, **kwargs):
    '''Kernel for calculating spectral densities from
       correlation of signals a (and b) with various options.
       '''
    # BETA: correlate moment of part with total moments, expects one integer
    sub = kwargs.get('subparticle')
    # BETA: correlate moment of one part with another part,
    #     expects tuple of integers
    #     notparticles: exclude these indices; expects list of integers
    subs = kwargs.get('subparticles')
    subnots = kwargs.get('subnotparticles')

    a = args[0]
    b = args[0]  # dummy
    cross = False
    if len(args) >= 2:
        b = args[1]
        cross = True

    def _get_spectra(_a, _b, cross, **kwargs):
        '''standard spectrum'''
        _a = _a.sum(axis=1)
        if not cross:
            omega, S_aa, R_aa = spectral_density(_a, **kwargs)
            return omega, S_aa, R_aa
        else:
            _b = _b.sum(axis=1)
            omega, S_ab, R_ab = spectral_density(_a, _b, **kwargs)
            return omega, S_ab, R_ab

    def _get_subspectra(_a, _b, cross, _sub1, _sub2=None, **kwargs):
        _a1 = _a[:, _sub1]
        if _sub2 is None:
            _a2 = _a.sum(axis=1)
        else:
            _a2 = _a[:, _sub2]
        if not cross:
            omega, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            return omega, S_aa, R_aa
        else:
            _b1 = _b[:, _sub1]
            if _sub2 is None:
                _b2 = _b.sum(axis=1)
            else:
                _b2 = _b[:, _sub2]
            omega, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            omega, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2
            return omega, S_ab, R_ab

    def _get_subnotspectra(_a, _b, cross, _subnot, **kwargs):
        _a1 = _a.sum(axis=1)
        _a2 = copy.deepcopy(_a)
        _a2[:, _subnot] = np.array([0.0, 0.0, 0.0])
        _a2 = _a2.sum(axis=1)
        if not cross:
            omega, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            return omega, S_aa, R_aa
        else:
            _b1 = _b.sum(axis=1)
            _b2 = copy.deepcopy(_b)
            _b2[:, _subnot] = np.array([0.0, 0.0, 0.0])
            _b2 = _b2.sum(axis=1)
            omega, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            omega, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2
            return omega, S_ab, R_ab

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

    return _get(a, b, cross, **kwargs)


def _spectrum_from_tcf(*args, **kwargs):
    '''Choose between modes: abs, cd, abs_cd
       Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)
           2 - magnetic dipole moments of shape
                (n_frames[, nkinds], three) (mode=cd).

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       No support of trajectory iterators.

       Expects atomic units (but it is not mandatory).

       ts ... timestep

       Returns dictionary with:
         "omega"            - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''
    mode = kwargs.get('mode', 'abs_cd')
    if mode not in ['abs', 'cd', 'abs_cd']:
        raise ValueError('Unknown mode', mode)
    _z = len(args)

    def _err(_s, _z):
        raise TypeError('tcf expected %d argument, got %d' % (_s, _z))
    if mode == 'abs':
        _s = 1
    else:
        _s = 2
    if _z != _s:
        _err(_s, _z)

    cur_dipoles = args[0]
    if 'cd' in mode:
        mag_dipoles = args[1]
        if mag_dipoles.shape != cur_dipoles.shape:
            raise ValueError('shapes of given data do not agree',
                             cur_dipoles.shape, mag_dipoles.shape)
    data = {}

    if len(cur_dipoles.shape) == 2:
        omega, _abs, C_abs = spectral_density(
                                cur_dipoles,
                                cur_dipoles,
                                **kwargs
                                )
        # _cc = constants.current_current_prefactor(300)
        _cc = 1.

        data['omega'] = omega
        data['abs'] = _abs * _cc
        data['tcf_abs'] = C_abs

    elif len(cur_dipoles.shape) == 3:
        positions = kwargs.get('positions')
        if positions is None:
            raise TypeError('Please give positions arguments for moments of '
                            'shape %s' % cur_dipoles.shape)
        origins = kwargs.get('origins', np.zeros((1, 1, 3)))
        cell = kwargs.get('cell')
        cutoff = kwargs.get('cutoff', 0)
        cut_type = kwargs.get('cut_type', 'soft')
        # BETA: remove direct correlation between background
        cutoff_bg = kwargs.get('background_correction_cutoff')

        spectrum = list()

        for _i, _o in enumerate(origins):
            _trans = distance_pbc(_o[:, None], positions, cell_aa_deg=cell)
            _c = copy.deepcopy(cur_dipoles)
            _c = _apply_cutoff(_c, _trans, cutoff, cut_type)
            if cutoff_bg is not None:
                _c_bg = copy.deepcopy(_c)
                _c_bg = _apply_cutoff(_c_bg, _trans, cutoff_bg, 'hard')

            if 'cd' in mode:
                _m = copy.deepcopy(mag_dipoles)
                # --- gauge-transport
                _m += magnetic_dipole_shift_origin(_c, _trans)
                _m = _apply_cutoff(_m, _trans, cutoff, cut_type)
                if cutoff_bg is not None:
                    _m_bg = copy.deepcopy(_m)
                    _m_bg = _apply_cutoff(_m_bg, _trans, cutoff_bg, 'hard')

            # --- get spectra
            # ToDo: sort out prefactors!
            _result = []
            if 'abs' in mode:
                omega, _abs, C_abs = _get_tcf_spectrum(_c, **kwargs)
                if cutoff_bg is not None:
                    _tmp = _get_tcf_spectrum(_c_bg, _c_bg, **kwargs)
                    _abs -= _tmp[1]
                    C_abs -= _tmp[2]
                # _cc = constants.current_current_prefactor(300)
                _cc = 1.
                _result += [_abs*_cc, C_abs]

            if 'cd' in mode:
                omega, _cd, C_cd = _get_tcf_spectrum(_c, _m, **kwargs)
                if cutoff_bg is not None:
                    _tmp = _get_tcf_spectrum(_c_bg, _m_bg, **kwargs)
                    _cd -= _tmp[1]
                    C_cd -= _tmp[2]
                # _cm = constants.current_magnetic_prefactor(omega, 300)
                _cm = 4. * omega
                _result += [_cd*_cm, C_cd]
            _result += [omega]

            spectrum.append(_result[::-1])

        # --- average over origins
        spectrum = np.array(spectrum).sum(axis=0) / origins.shape[0]

        data['omega'] = _result.pop()

        if 'cd' in mode:
            data['tcf_cd'] = _result.pop()
            data['cd'] = _result.pop()

        if 'abs' in mode:
            data['tcf_abs'] = _result.pop()
            data['abs'] = _result.pop()

        # ToDo: warnings.warn("Return tcf incomplete!", stacklevel=2)
    else:
        raise TypeError('data with wrong shape!',
                        cur_dipoles.shape)

    return data


def absorption_from_tcf(*args, **kwargs):
    '''Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       No support of trajectory iterators.

       Expects atomic units (but it is not mandatory).

       ts ... timestep

       Returns dictionary with:
         "omega"       - discrete sample frequencies
         "abs"         - spectral density (FT TCF)
         "tcf_abs"     - time-correlation function (TCF)
       '''
    kwargs.update({'mode': 'abs'})
    return _spectrum_from_tcf(*args, **kwargs)


def circular_dichroism_from_tcf(*args, **kwargs):
    '''Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)
           2 - magnetic dipole moments of shape
                (n_frames[, nkinds], three) (mode=cd).

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       No support of trajectory iterators.

       Expects atomic units (but it is not mandatory).

       ts ... timestep

       Returns dictionary with:
         "omega"      - discrete sample frequencies
         "cd"         - spectral density (FT TCF)
         "tcf_cd"     - time-correlation function (TCF)
       '''
    kwargs.update({'mode': 'cd'})
    return _spectrum_from_tcf(*args, **kwargs)
