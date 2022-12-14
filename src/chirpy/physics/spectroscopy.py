# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2020-2022, The ChirPy Developers.
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
# ----------------------------------------------------------------------

import copy
import numpy as np
import warnings as _warnings
import tqdm

from .classical_electrodynamics import magnetic_dipole_shift_origin
from .statistical_mechanics import spectral_density
from .. import constants
from ..classes.object import Sphere
from ..classes.core import PALARRAY
from ..topology.mapping import distance_pbc, _pbc_shift
from ..topology import mapping

from ..config import ChirPyWarning as _ChirPyWarning


def absorption_from_transition_moment(etdm_au):
    '''Integrated absorption coeffcient in atomic units corresponding to
       distance**2 / time / amount.
       For integration over wavenumbers, i.e. distance / amount
       (e.g., km / mol in SI), divide by speed of light and transform
       units accordingly.
       etdm_au:    electric transition dipole moment of a normal mode
                   in atomic units (charge / sqrt(mass))
                   NB: assuming mass-weighted coordinate distance * sqrt(mass)!
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


def power_from_tcf(velocities_au, weights=1.0,
                   ts_au=41.341,
                   average_atoms=True,
                   window_length_au=None,
                   flt_pow=None,
                   **kwargs):
    '''Expects velocities of shape (n_frames, n_atoms, three)
       No support of trajectory iterators.

       Expects atomic units.

       ts_au ... timestep in a.u.
       weights ... atom weights (in most cases: atom masses)
       window_length_au ... time length of the window function used with the
                            time correlation function in a.u.

       The output is averaged over the no. of atoms/species.
       Returns dictionary with (all in a.u.):
         "f"             - discrete sample frequencies
         "power"         - spectral density (FT TCF) in <energy>
                           (for weights in <mass>)
         "tcf_power"     - time-correlation function (TCF)
       '''
    if flt_pow is not None:
        _warnings.warn('The flt_pow keyword is deprecated, '
                       'use window_length_au instead',
                       _ChirPyWarning, stacklevel=2)

        kwargs.update({'flt_pow': flt_pow})

        if flt_pow >= 0:
            _warnings.warn('Got non-negative value for flt_pow; FT-TCF spectra'
                           ' require flt_pow < 0 to account for finite size of'
                           ' input data!', _ChirPyWarning, stacklevel=2)

    if 'ts' in kwargs:
        raise KeyError('ts argument must not be used here. Please specify '
                       'ts_au!')
    kwargs.update(dict(ts=ts_au))
    if window_length_au is not None:
        kwargs.update(dict(window_length=int(window_length_au/ts_au)))

    n_frames, n_atoms, n_dims = velocities_au.shape

    if not hasattr(weights, '__len__'):
        wgh = np.ones(n_atoms * n_dims) * weights
    else:
        wgh = np.repeat(weights, n_dims)

    _velocities = velocities_au.reshape((n_frames, -1))
    f, S, R = zip(*[spectral_density(_v, symmetry='even', **kwargs)
                    for _v in _velocities.T])

    data = {}
    data['freq'] = f[0]
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

       ts_au ... timestep in a.u.
       window_length_au ... time length of the window function used with the
                            time correlation function in a.u.

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
       window_length_au ... time length of the window function used with the
                            time correlation function in a.u.

       Returns dictionary with (all in a.u.):
         "freq"             - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF) in
                              distance**2 = 1 / (distance * density)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''
    kwargs.update({'mode': 'cd'})
    return _spectrum_from_tcf(*args, **kwargs)


def _apply_cut_sphere(x, pos, clip, cell=None, inverse=False):
    if len(clip) != 0:
        y = np.zeros_like(x)
        for _tr in clip:
            if _tr.__class__.__name__ != 'Sphere':
                raise TypeError('list does not contain ChirPy Spheres.')
            y += _tr.clip_section_observable(np.ones_like(x),
                                             pos,
                                             cell=cell,
                                             inverse=inverse)
    else:
        y = np.ones_like(x)
    return np.clip(y, 0, 1) * x


def _spectrum_from_tcf(*args, **kwargs):
    return spectrum_from_tcf(*args, **kwargs)


def spectrum_from_tcf(*args,
                      T_K=300,
                      mode='abs_cd',
                      origin_au=np.zeros((3)),
                      cell_au_deg=None,
                      positions_au=None,
                      gauge_transport=True,
                      pseudo_isolated=False,
                      cutoff_au=None,
                      cutoff_bg_au=None,
                      cut_type='hard',
                      cut_type_bg='hard',
                      clip_sphere=[],
                      flt_pow=None,
                      window_length_au=None,
                      ts_au=41.341,
                      unwrap_pbc=True,
                      parallel=True,
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

       Expects atomic units.

       ts_au ... timestep in a.u.
       window_length_au ... if not None, apply window function (Welch)
                            of the given time length in a.u. to the
                            time correlation function

       Computation of the gauge transport:
         unwrap_pbc ... Unwrap particles before the calculation.
         parallel ... Execute job in parallel (PALARRAY).
         pseudo_isolated ... Impose isolated supercell with origin_au as
                             centre. This option accelerates the computation
                             and is applicable especially to large supercells
                             (e.g., as solvent bath).
                             However, the inequality
                                 cutoff <= ||cell_vector|| / 4.
                             has to be fulfilled.


       Returns dictionary with (all in a.u.):
         "freq"             - discrete sample frequencies
         "abs"/"cd"         - spectral density (FT TCF) in
                              <distance**2> = 1 / (<distance> * <density>)
         "tcf_abs"/"tcf_cd" - time-correlation function (TCF)
       '''

    if flt_pow is not None:
        _warnings.warn('The flt_pow keyword is deprecated, '
                       'use window_length_au instead',
                       _ChirPyWarning, stacklevel=2)

        kwargs.update({'flt_pow': flt_pow})

        if flt_pow >= 0:
            _warnings.warn('Got non-negative value for flt_pow; FT-TCF spectra'
                           ' require flt_pow < 0 to account for finite size of'
                           ' input data!', _ChirPyWarning, stacklevel=2)

    if 'ts' in kwargs:
        raise KeyError('ts argument must not be used here. Please specify '
                       'ts_au!')
    kwargs.update(dict(ts=ts_au))
    kwargs.update(dict(symmetry='even'))
    if window_length_au is not None:
        kwargs.update(dict(window_length=int(window_length_au/ts_au)))

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

    # --- end of argument parsing
    data = {}

    if len(cur_dipoles.shape) == 2:
        if 'abs' in mode:
            freq, _abs, C_abs = spectral_density(
                                    cur_dipoles,
                                    cur_dipoles,
                                    **kwargs
                                    )
            data['freq'] = freq
            data['abs'] = _abs
            data['tcf_abs'] = C_abs

        if 'cd' in mode:
            # --- NB: no gauge-transport here!
            freq, _cd, C_cd = spectral_density(
                                    cur_dipoles,
                                    mag_dipoles,
                                    **kwargs
                                    )
            data['cd'] = _cd
            data['tcf_cd'] = C_cd

    elif len(cur_dipoles.shape) == 3:
        if positions_au is None:
            raise TypeError('Please give positions_au argument for moments of '
                            f'shape {cur_dipoles.shape}')

        # --- map origin on frames if frame dim is missing
        if len(origin_au.shape) == 1:
            origin_au = np.tile(origin_au, (positions_au.shape[0], 1))

        pos = copy.deepcopy(positions_au)
        cell = cell_au_deg

        # --- cutoff spheres --------------------------------------------------
        if not isinstance(clip_sphere, list):
            raise TypeError('expected list for keyword "clip_sphere"!')

        # --- master sphere (cutoff) ==> applied ON TOP OF clip spheres
        _cut_sphere = []
        if cutoff_au is not None:
            _cut_sphere.append(Sphere(
                                 origin_au,
                                 cutoff_au,
                                 edge=cut_type
                                 ))
            pseudo_isolated = True
        # ---------------------------------------------------------------------

        _c = copy.deepcopy(cur_dipoles)
        # --- apply handed over spheres
        _c = _apply_cut_sphere(_c, pos, clip_sphere, cell=cell)
        # --- apply general cutoff
        _c = _apply_cut_sphere(_c, pos, _cut_sphere, cell=cell)
        if 'cd' in mode:
            _m = copy.deepcopy(mag_dipoles)
            _m = _apply_cut_sphere(_m, pos, clip_sphere, cell=cell)
            _m = _apply_cut_sphere(_m, pos, _cut_sphere, cell=cell)

        # --- assume non-periodic boundaries ?
        if pseudo_isolated or cell_au_deg is None:
            # _warnings.warn('imposing non-periodic boundaries around '
            #                f'origin(s) {origin_au}',
            #                _ChirPyWarning,
            #                stacklevel=2)
            pos = distance_pbc(origin_au[:, None],
                               positions_au,
                               cell_au_deg)
            origin_au = np.zeros_like(origin_au)
            cell = None

        # --- get spectra
        data['c'] = _c
        if 'abs' in mode:
            freq, _abs, C_abs = spectral_density(_c.sum(axis=1), **kwargs)
            data['tcf_abs'] = C_abs
            data['abs'] = _abs

        if 'cd' in mode:
            freq, _cd, C_cd = spectral_density(_c.sum(axis=1), _m.sum(axis=1),
                                               **kwargs)
            data['m'] = _m
            data['tcf_cd'] = C_cd
            data['cd'] = _cd

            if gauge_transport:
                _cd_GT, C_cd_GT = compute_gauge_transport_term(
                                                       _c, pos, cell,
                                                       unwrap_pbc=unwrap_pbc,
                                                       parallel=parallel,
                                                       **kwargs)
                data['cd'] += _cd_GT
                data['tcf_cd'] += C_cd_GT
            else:
                _warnings.warn('Omitting gauge transport term in CD mode!',
                               _ChirPyWarning, stacklevel=2)

        data['freq'] = freq

        if cutoff_bg_au is not None:
            data = _background_correction(
                               data,
                               pos,
                               origin_au,
                               cutoff_bg_au,
                               cut_type_bg,
                               cell_au_deg=cell,
                               gauge_transport=gauge_transport,
                               unwrap_pbc=unwrap_pbc,
                               parallel=parallel,
                               **kwargs)
    else:
        raise TypeError('data with wrong shape!', cur_dipoles.shape)

    if 'abs' in data:
        data['abs'] *= constants.current_current_prefactor_au(T_K)
    if 'cd' in data:
        data['cd'] *= constants.current_magnetic_prefactor_au(T_K,
                                                              freq*2*np.pi)

    return data


def gauge_transport_particle_i(_i, pos, cur, cell, **kwargs) -> tuple:
    '''_i ... index of particle
       pos/cur ... full arrays of shape (n_frames, n_particles, 3)
       '''
    n_frames, n_particles, n_dim = pos.shape
    a = cur[:, _i]

    if np.abs(a).sum() > 1.E-16:
        if cell is not None:
            _delta = -_pbc_shift(pos - pos[:, _i, None], cell)
        else:
            _delta = np.zeros_like(pos)

        b = -0.5 * magnetic_dipole_shift_origin(
                                    cur,
                                    2 * pos + _delta
                                    ).sum(axis=1)
        return spectral_density(a, b, **kwargs)
    else:
        return np.zeros(n_frames), np.zeros(n_frames), np.zeros(2*n_frames-1)


def compute_gauge_transport_term(cur, pos, cell,
                                 unwrap_pbc=True,
                                 parallel=True,
                                 **kwargs):
    '''
       unwrap_pbc ... unwrap particles before the calculation
       parallel ... execute job in parallel (PALARRAY)
    '''
    n_frames, n_particles, n_dim = pos.shape

    if cell is None:
        a = cur.sum(axis=1)
        b = -magnetic_dipole_shift_origin(cur, pos).sum(axis=1)

        return spectral_density(a, b, **kwargs)[1:]

    _pos = copy.deepcopy(pos)
    if unwrap_pbc:
        _pos = mapping.unwrap_pbc(pos, cell=cell)

    if not parallel:
        freq, cd_GT, C_cd_GT = np.array([
                     gauge_transport_particle_i(
                      _i,
                      pos=_pos,
                      cur=cur,
                      cell=cell,
                      **kwargs) for _i in tqdm.tqdm(
                                         range(n_particles),
                                         desc='gauge_transport_particle_i'
                                         )],
                     dtype='object'
                     ).swapaxes(0, 1)

    else:
        freq, cd_GT, C_cd_GT = tuple(zip(*PALARRAY(
                      gauge_transport_particle_i,
                      range(n_particles),
                      pos=_pos,
                      cur=cur,
                      cell=cell,
                      **kwargs
                      ).run()))  # .swapaxes(0, 1)

    return np.array(cd_GT).sum(axis=0), np.array(C_cd_GT).sum(axis=0)


def _background_correction(data, pos_au, origin_au, cutoff_bg_au, cut_type_bg,
                           cell_au_deg=None,
                           gauge_transport=True,
                           unwrap_pbc=True,
                           parallel=True,
                           **kwargs):
    '''Compute spectral density outside a given background cutoff'''
    _cut_sphere_bg = [Sphere(origin_au, cutoff_bg_au, edge=cut_type_bg)]
    _c_bg = _apply_cut_sphere(copy.deepcopy(data['c']), pos_au, _cut_sphere_bg,
                              inverse=True, cell=cell_au_deg)
    if 'abs' in data:
        _tmp = spectral_density(_c_bg.sum(axis=1), **kwargs)
        data['abs'] -= _tmp[1]
        data['tcf_abs'] -= _tmp[2]
        data['c_bg'] = _c_bg

    if 'cd' in data:
        _m_bg = _apply_cut_sphere(copy.deepcopy(data['m']), pos_au,
                                  _cut_sphere_bg,
                                  inverse=True, cell=cell_au_deg)
        _tmp = spectral_density(_c_bg.sum(axis=1), _m_bg.sum(axis=1), **kwargs)
        data['cd'] -= _tmp[1]
        data['tcf_cd'] -= _tmp[2]
        data['m_bg'] = _m_bg
        if gauge_transport:
            _cd_GT, C_cd_GT = compute_gauge_transport_term(
                                                        _c_bg,
                                                        pos_au,
                                                        cell_au_deg,
                                                        unwrap_pbc=unwrap_pbc,
                                                        parallel=parallel,
                                                        **kwargs
                                                        )
            data['cd'] -= _cd_GT
            data['tcf_cd'] -= C_cd_GT

    return data


# --- legacy code
# def _get_tcf_spectrum(*args,
#                       subparticle0=None,
#                       subparticle1=None,
#                       subnotparticles=None,
#                       **kwargs):
#     '''Kernel for calculating spectral densities from
#        correlation of signals a (and b) with various options.
#        '''
#     # --- correlate moment of part with total moments, expects one integer
#     sub0 = subparticle0
#     sub1 = subparticle1
#     if sub1 is not None and sub0 is None:
#         sub0 = sub1
#         sub1 = None
#
#     # --- exclude these indices; expects list of integers
#     subnots = subnotparticles
#
#     a = args[0]
#     b = args[0]  # dummy
#     cross = False
#     if len(args) >= 2:
#         b = args[1]
#         cross = True
#
#     def _get_spectra(_a, _b, cross, **kwargs):
#         '''standard spectrum'''
#         _a = _a.sum(axis=1)
#         if not cross:
#             f, S_aa, R_aa = spectral_density(_a, **kwargs)
#             return f, S_aa, R_aa
#         else:
#             _b = _b.sum(axis=1)
#             f, S_ab, R_ab = spectral_density(_a, _b, **kwargs)
#             return f, S_ab, R_ab
#
#     def _get_subspectra(_a, _b, cross, _sub0, _sub1=None, **kwargs):
#         _a1 = _a[:, _sub0]
#         if _sub1 is None:
#             _a2 = _a.sum(axis=1)
#         else:
#             _a2 = _a[:, _sub1]
#         if not cross:
#             f, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
#             return f, S_aa, R_aa
#         else:
#             _b1 = _b[:, _sub0]
#             if _sub1 is None:
#                 _b2 = _b.sum(axis=1)
#             else:
#                 _b2 = _b[:, _sub1]
#             f, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
#             f, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
#             S_ab = (S_ab1 + S_ab2) / 2
#             R_ab = (R_ab1 + R_ab2) / 2
#             return f, S_ab, R_ab
#
#     def _get_subnotspectra(_a, _b, cross, _subnot, **kwargs):
#         _a1 = _a.sum(axis=1)
#         _a2 = copy.deepcopy(_a)
#         _a2[:, _subnot] = np.array([0.0, 0.0, 0.0])
#         _a2 = _a2.sum(axis=1)
#         if not cross:
#             f, S_aa, R_aa = spectral_density(_a1, _a2, **kwargs)
#             return f, S_aa, R_aa
#         else:
#             _b1 = _b.sum(axis=1)
#             _b2 = copy.deepcopy(_b)
#             _b2[:, _subnot] = np.array([0.0, 0.0, 0.0])
#             _b2 = _b2.sum(axis=1)
#             f, S_ab1, R_ab1 = spectral_density(_a1, _b2, **kwargs)
#             f, S_ab2, R_ab2 = spectral_density(_a2, _b1, **kwargs)
#             S_ab = (S_ab1 + S_ab2) / 2
#             R_ab = (R_ab1 + R_ab2) / 2
#             return f, S_ab, R_ab
#
#     if sub0 is not None:
#         if not isinstance(sub0, int):
#             raise TypeError('Subparticle has to be an integer!')
#         if sub1 is not None:
#             if not isinstance(sub1, int):
#                 raise TypeError('Subparticle has to be an integer!')
#             _get = partial(_get_subspectra, _sub0=sub0, _sub1=sub1)
#
#         else:
#             _get = partial(_get_subspectra, _sub0=sub0)
#
#     elif subnots is not None:
#         if not isinstance(subnots, list):
#             raise TypeError('Subnotparticles has to be a list!')
#         _get = partial(_get_subnotspectra, _subnot=subnots)
#
#     else:
#         _get = _get_spectra
#
#     return _get(a, b, cross, **kwargs)
