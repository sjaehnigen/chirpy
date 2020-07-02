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
import numpy as np
from functools import partial
import warnings as _warnings

from .classical_electrodynamics import switch_magnetic_origin_gauge
from .statistical_mechanics import spectral_density
from ..physics import constants
from ..classes.object import Sphere


def absorption_from_transition_moment(etdm_au):
    '''Integrated absorption coeffcient in atomic units corresponding to
       distance**2 / time / amount.
       For integration over wavenumbers, i.e. distance / amount
       (e.g., km / mol in SI), divide by speed of light and transform
       units accordingly.
       etdm_au:     electric transition dipole moment in atomic units
                    (charge)
                    NB: NO mass-weighted coordinate charge / sqrt(mass)!
       '''
    # --- see Neugebauer2002
    # --- we take the prefacor from Fermi's Golden Rule and combine it with the
    #     harmonic oscillator approximation (prefactors); dependencies on omega
    #     and the temperature cancel so we use a dummy values
    T = 100
    w = 1
    prefactor_au = constants.dipole_dipole_prefactor_au(T, w)
    prefactor_au *= T * constants.k_B_au / w**2
    # --- from omega to nu
    prefactor_au /= 2 * np.pi
    # --- average intensity over period
    prefactor_au /= 2

    return (etdm_au ** 2).sum(axis=-1) * prefactor_au


def circular_dichroism_from_transition_moments(etdm_au, mtdm_au):
    '''Integrated differential absorption coeffcient in atomic units
       corresponding to distance**2 / time / amount.
       For integration over wavenumbers, i.e. distance / amount
       (e.g., km / mol in SI), divide by speed of light and transform
       units accordingly.
       etdm_au:     electric transition dipole moment in atomic units
                    (charge)
       mtdm_au:     magnetic transition dipole moment in atomic units
                    (current * distance)
                    NB: NO mass-weighted coordinate * 1 / sqrt(mass)!
                    NB: NO cgs-convention for magnetic moments!
       '''
    # --- see Neugebauer2002
    # --- we take the prefacor from Fermi's Golden Rule and combine it with the
    #     harmonic oscillator approximation (prefactors); dependencies on omega
    #     and the temperature cancel so we use a dummy values
    T = 100
    w = 1
    prefactor_au = constants.dipole_magnetic_prefactor_au(T, w)
    prefactor_au *= T * constants.k_B_au / w**2
    # --- from omega to nu
    prefactor_au /= 2 * np.pi
    # --- average intensity over period
    prefactor_au /= 2

    return (etdm_au * mtdm_au).sum(axis=-1) * prefactor_au


def power_from_tcf(velocities, weights=1.0,
                   flt_pow=-1.E-99,
                   average_atoms=True, **kwargs):
    '''Expects velocities of shape (n_frames, n_atoms, three)
       No support of trajectory iterators.

       Expects atomic units for the correct prefactors.

       ts ... timestep in a.u.

       The output is averaged over the no. of atoms/species.
       Returns dictionary with (all in a.u.):
         "f"             - discrete sample frequencies
         "power"         - spectral density (FT TCF) in <energy>
                           (for weights in <mass>)
         "tcf_power"     - time-correlation function (TCF)
       '''
    if flt_pow >= 0:
        _warnings.warn('Got non-negative value for flt_pow; FT-TCF spectra '
                       'require flt_pow < 0 to account for finite size of '
                       'input data!', stacklevel=2)
    kwargs.update({'flt_pow': flt_pow})
    n_frames, n_atoms, n_dims = velocities.shape

    if not hasattr(weights, '__len__'):
        wgh = np.ones(n_atoms * n_dims) * weights
    else:
        wgh = np.repeat(weights, n_dims)

    _velocities = velocities.reshape((n_frames, -1))
    f, S, R = zip(*[spectral_density(_v, **kwargs)
                    for _v in _velocities.T])

    data = {}
    data['f'] = f[0]
    if average_atoms:
        data['power'] = np.mean(np.array(S) * wgh[:, None], axis=0)
        data['tcf_power'] = np.mean(np.array(R), axis=0)
    else:
        data['power'] = np.array(S) * wgh[:, None]
        data['tcf_power'] = np.array(R)

    return data


def absorption_from_tcf(*args, **kwargs):
    '''Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       No support of trajectory iterators.

       Expects atomic units for the correct prefactors.

       ts ... timestep in a.u.

       Returns dictionary with (all in a.u.):
         "freq"             - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF) in
                              distance**2 = 1 / (distance * density)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''
    kwargs.update({'mode': 'abs'})
    return _spectrum_from_tcf(*args, **kwargs)


def circular_dichroism_from_tcf(*args, **kwargs):
    '''Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)
           2 - magnetic dipole moments of shape
                (n_frames[, nkinds], three) (mode=cd).
           NB: No cgs-convention for magnetic properties!

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       No support of trajectory iterators.

       Expects atomic units for the correct prefactors.

       ts ... timestep in a.u.

       Returns dictionary with (all in a.u.):
         "freq"             - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF) in
                              distance**2 = 1 / (distance * density)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''
    kwargs.update({'mode': 'cd'})
    return _spectrum_from_tcf(*args, **kwargs)


def _spectrum_from_tcf(*args,
                       T_K=300,
                       mode='abs_cd',
                       origin=np.zeros((3)),
                       cell_au_deg=None,
                       positions=None,
                       return_moments=False,
                       gauge_transport=True,
                       cutoff=None,
                       cutoff_bg=None,
                       cut_type='soft',
                       cut_type_bg='hard',
                       clip_sphere=[],
                       flt_pow=-1.E-99,
                       **kwargs):
    '''Choose between modes: abs, cd, abs_cd
       Expects
           1 - current (electric) dipole moments of shape
               (n_frames[, nkinds], three) (mode=abs)
           2 - magnetic dipole moments of shape
                (n_frames[, nkinds], three) (mode=cd).

       When specifying multiple kinds (2nd dimension), an
       additional named argument ("positions") containing
       the positions of kinds of the same shape is required.
       Specify the gauge origin with origin= (optional)
       No support of trajectory iterators.

       Expects atomic units (but it is not mandatory).

       ts ... timestep in a.u.

       Returns dictionary with (all in a.u.):
         "freq"             - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF) in
                              <distance**2> = 1 / (<distance> * <density>)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''

    if flt_pow >= 0:
        _warnings.warn('Got non-negative value for flt_pow; FT-TCF spectra '
                       'require flt_pow < 0 to account for finite size of '
                       'input data!', stacklevel=2)
    kwargs.update({'flt_pow': flt_pow})
    cell = cell_au_deg
    r_moments = return_moments
    pos = positions

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
        freq, _abs, C_abs = spectral_density(
                                cur_dipoles,
                                cur_dipoles,
                                **kwargs
                                )

        _cc = constants.current_current_prefactor_au(T_K)

        if 'cd' in mode:
            # --- NB: no gauge-transport here!
            freq, _cd, C_cd = spectral_density(
                                    cur_dipoles,
                                    mag_dipoles,
                                    **kwargs
                                    )
            _cm = constants.current_magnetic_prefactor_au(T_K, freq*2*np.pi)
            data['cd'] = _cd * _cm
            data['tcf_cd'] = C_cd

        data['freq'] = freq
        data['abs'] = _abs * _cc
        data['tcf_abs'] = C_abs

    elif len(cur_dipoles.shape) == 3:
        if pos is None:
            raise TypeError('Please give positions arguments for moments of '
                            f'shape {cur_dipoles.shape}')

        # --- map origin on frames if frame dim is missing
        if len(origin.shape) == 1:
            origin = np.tile(origin, (pos.shape[0], 1))

        # --- cutoff spheres --------------------------------------------------
        if not isinstance(clip_sphere, list):
            raise TypeError('expected list for keyword "clip_sphere%s"!')

        # --- master sphere (cutoff) ==> applied ON TOP OF clip spheres
        _cut_sphere = []
        if cutoff is not None:
            _cut_sphere.append(Sphere(
                                 origin,
                                 cutoff,
                                 edge=cut_type
                                 ))

        _cut_sphere_bg = []
        if cutoff_bg is not None:
            _cut_sphere_bg.append(Sphere(
                                    origin,
                                    cutoff_bg,
                                    edge=cut_type_bg
                                    ))
        # ---------------------------------------------------------------------

        def _cut(x, pos, clip, inverse=False):
            if len(clip) != 0:
                y = np.zeros_like(x)
                for _tr in clip:
                    if not isinstance(_tr, Sphere):
                        raise TypeError('expected a list of Sphere objects as '
                                        'clip spheres!')
                    y += _tr.clip_section_observable(np.ones_like(x),
                                                     pos,
                                                     cell=cell,
                                                     inverse=inverse)
            else:
                y = np.ones_like(x)
            return np.clip(y, 0, 1) * x

        _c = copy.deepcopy(cur_dipoles)
        _c = _cut(_c, pos, clip_sphere)
        _c = _cut(_c, pos, _cut_sphere)
        if 'cd' in mode:
            _m = copy.deepcopy(mag_dipoles)
            _m = _cut(_m, pos, clip_sphere)
            _m = _cut(_m, pos, _cut_sphere)

            # --- calculate gauge-transport
            if gauge_transport:
                _m = switch_magnetic_origin_gauge(_c, _m, pos, origin[:, None],
                                                  cell_au_deg=cell)
            else:
                _warnings.warn('Omitting gauge transport term in CD mode!',
                               stacklevel=2)

        if len(_cut_sphere_bg) != 0:
            _c_bg = copy.deepcopy(_c)
            _c_bg = _cut(_c_bg, pos, _cut_sphere_bg, inverse=True)
            if 'cd' in mode:
                _m_bg = copy.deepcopy(_m)
                _m_bg = _cut(_m_bg, pos, _cut_sphere_bg, inverse=True)

        # --- get spectra
        _result = []
        if 'abs' in mode:
            freq, _abs, C_abs = _get_tcf_spectrum(_c, **kwargs)
            _cc = constants.current_current_prefactor_au(T_K)
            if len(_cut_sphere_bg) != 0:
                _tmp = _get_tcf_spectrum(_c_bg, _c_bg, **kwargs)
                _abs -= _tmp[1]
                C_abs -= _tmp[2]
            _result += [_abs*_cc, C_abs]

        if 'cd' in mode:
            freq, _cd, C_cd = _get_tcf_spectrum(_c, _m, **kwargs)
            _cm = constants.current_magnetic_prefactor_au(T_K, freq*2*np.pi)
            if len(_cut_sphere_bg) != 0:
                _tmp = _get_tcf_spectrum(_c_bg, _m_bg, **kwargs)
                _cd -= _tmp[1]
                C_cd -= _tmp[2]
            _result += [_cd*_cm, C_cd]
        _result += [freq]

        data['freq'] = _result.pop()

        if 'cd' in mode:
            data['tcf_cd'] = _result.pop()
            data['cd'] = _result.pop()

        if 'abs' in mode:
            data['tcf_abs'] = _result.pop()
            data['abs'] = _result.pop()

        # --- write moments to dictionary (optional)
        if r_moments:
            data['c'] = _c
            if 'cd' in mode:
                data['m'] = _m
            if len(_cut_sphere_bg) != 0:
                data['c_bg'] = _c_bg
                if 'cd' in mode:
                    data['m_bg'] = _m_bg

    else:
        raise TypeError('data with wrong shape!',
                        cur_dipoles.shape)

    return data


def _get_tcf_spectrum(*args,
                      subparticle0=None,
                      subparticle1=None,
                      subnotparticles=None,
                      **kwargs):
    '''Kernel for calculating spectral densities from
       correlation of signals a (and b) with various options.
       '''
    # --- correlate moment of part with total moments, expects one integer
    sub0 = subparticle0
    sub1 = subparticle1
    if sub1 is not None and sub0 is None:
        sub0 = sub1
        sub1 = None

    # --- exclude these indices; expects list of integers
    subnots = subnotparticles

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
            f, S_aa, R_aa = spectral_density(_a, **kwargs)
            return f, S_aa, R_aa
        else:
            _b = _b.sum(axis=1)
            f, S_ab, R_ab = spectral_density(_a, _b, **kwargs)
            return f, S_ab, R_ab

    def _get_subspectra(_a, _b, cross, _sub0, _sub1=None, **kwargs):
        _a1 = _a[:, _sub0]
        if _sub1 is None:
            _a2 = _a.sum(axis=1)
        else:
            _a2 = _a[:, _sub1]
        if not cross:
            f, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            return f, S_aa, R_aa
        else:
            _b1 = _b[:, _sub0]
            if _sub1 is None:
                _b2 = _b.sum(axis=1)
            else:
                _b2 = _b[:, _sub1]
            f, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            f, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2
            return f, S_ab, R_ab

    def _get_subnotspectra(_a, _b, cross, _subnot, **kwargs):
        _a1 = _a.sum(axis=1)
        _a2 = copy.deepcopy(_a)
        _a2[:, _subnot] = np.array([0.0, 0.0, 0.0])
        _a2 = _a2.sum(axis=1)
        if not cross:
            f, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
            return f, S_aa, R_aa
        else:
            _b1 = _b.sum(axis=1)
            _b2 = copy.deepcopy(_b)
            _b2[:, _subnot] = np.array([0.0, 0.0, 0.0])
            _b2 = _b2.sum(axis=1)
            f, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
            f, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
            S_ab = (S_ab1 + S_ab2) / 2
            R_ab = (R_ab1 + R_ab2) / 2
            return f, S_ab, R_ab

    if sub0 is not None:
        if not isinstance(sub0, int):
            raise TypeError('Subparticle has to be an integer!')
        if sub1 is not None:
            if not isinstance(sub1, int):
                raise TypeError('Subparticle has to be an integer!')
            _get = partial(_get_subspectra, _sub0=sub0, _sub1=sub1)

        else:
            _get = partial(_get_subspectra, _sub0=sub0)

    elif subnots is not None:
        if not isinstance(subnots, list):
            raise TypeError('Subnotparticles has to be a list!')
        _get = partial(_get_subnotspectra, _subnot=subnots)

    else:
        _get = _get_spectra

    return _get(a, b, cross, **kwargs)
