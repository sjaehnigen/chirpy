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
from ..physics import constants


def kinetic_energies(vel_au, masses_amu):
    """Calculate kinetic energy in a.u. from given velocities in a.u.
       and masses in a.m.u.
       velocities of shape ([n_frames,] n_atoms, three)"""

    e_kin_au = np.square(vel_au).sum(axis=-1) * masses_amu
    return 0.5 * e_kin_au * constants.m_amu_au


def temperature_from_energies(e_kin_au, fixed_dof=6):
    '''Ideal gas after Boltzmann.
       fixed_dof=6 ... fixed total linear and angular momentum.
       Expects kinetic energies of shape ([n_frames,] n_atoms)'''
    _n_dof = 3 * e_kin_au.shape[-1] - fixed_dof
    return (2. * e_kin_au / constants.k_B_au / _n_dof).sum(axis=-1)


def maxwell_boltzmann_distribution(T_K, *args, option='energy'):
    '''Return the Maxwell-Boltzmann distribution function for given temperature
       in K and species with masses in a.m.u.'''

    def _velocity_distribution(T_K, vel_norm_au, mass_amu):
        '''Returns the probability density of a given velocity in a.u. of a
           particle with mass in a.m.u. at a given temperature in K.
           Accepts np.arrays of species if shapes of vel and mass are equal.'''
        m_au = mass_amu * constants.m_amu_au
        beta = constants.k_B_au * T_K

        N = np.sqrt(2 / np.pi) * pow(m_au / beta, 3.0/2.0)
        p1 = np.exp(-(m_au * vel_norm_au**2) / (2 * beta))

        return N * p1 * vel_norm_au**2

    def _energy_distribution(T_K, E_au):
        '''Returns the probability density of a given energy in a.u.
           at a given temperature in K'''
        beta = constants.k_B_au * T_K

        N = 2 * np.sqrt(E_au / np.pi) * pow(1 / beta, 3.0/2.0)
        p1 = np.exp(-E_au / beta)

        return N * p1

    _options = {
            'velocity': _velocity_distribution,
            'energy': _energy_distribution,
            }

    def PDF(x):
        return _options.get(option)(T_K, x, *args)

    return PDF


def signal_filter(n_frames, filter_length=None, filter_type='welch'):
    if filter_length is None:
        filter_length = n_frames
    if filter_type == 'hanning':
        return np.hanning(2 * filter_length)[n_frames:]
    elif filter_type == 'welch':
        return (np.arange(filter_length)[::-1]/(filter_length+1))**2
    elif filter_type == 'triangular':
        return np.ones(filter_length) - np.arange(filter_length)/n_frames
    else:
        raise Exception('Filter %s not supported!' % filter_type)


def time_correlation_function(*args, flt_pow=0, cc_mode='AB', sum_dims=True):
    '''Calculate the time-correlation function (TCF) of a signal and
       using the Wiener-Khinchin theorem (fftconvolve).
       The method automatically chooses to calculate auto- or cross-
       correlation functions based on the number of arguments (max 2).
       Adding signal filters (Welch) may be enabled; use flt_pow=-1 to remove
       the implicit triangular filter due to finite size.
       Expects signal of shape (n_frames, n_dim)
       sum … sum over <n_dim> dimensions
       Returns:
        1 - time-correlation function (timestep as in input)
       '''

    if len(args) == 1:
        # --- auto-correlation
        val1 = args[0]
        val2 = args[0]

    elif len(args) == 2:
        # --- cross-correlation
        val1 = args[0]
        val2 = args[1]

    else:
        raise TypeError('TCF takes at most 2 arguments, got %d'
                        % len(args))

    _sh1 = val1.shape
    _sh2 = val2.shape
    if _sh1 != _sh2:
        raise ValueError('Got different shapes for signals val1 and val2: '
                         '%s and %s' % (_sh1, _sh2))

    if len(_sh1) == 1:
        n_frames, = _sh1
        val1 = val1.reshape((n_frames, 1))
        val2 = val2.reshape((n_frames, 1))

    elif len(_sh1) == 2:
        n_frames, n_dim = val1.shape
    else:
        raise ValueError('Expected shape length 1 or 2 for signal, got %d: %s'
                         % (len(_sh1), _sh1))

    def _corr(_val1, _val2, cc_mode='AB'):
        _sig = np.array([np.correlate(
                                  v1,
                                  v2,
                                  mode='full',
                                  )
                         for v1, v2 in zip(_val1.T, _val2.T)]).T

        R = np.zeros_like(_val1)
        if 'A' in cc_mode or 'C' in cc_mode:
            R += _sig[n_frames-1:]
        if 'B' in cc_mode:
            R += _sig[:n_frames][::-1]
        if 'C' in cc_mode:
            R -= _sig[:n_frames][::-1]
        if cc_mode == 'AB':
            R = R / 2.

        return R

    # cc mode:
    #   A or B = Ra or Rb
    #   AB = (Ra + Rb) / 2
    #   AC = Ra - Rb (benchmark)
    #   BC = 0
    #   ABC = A
    # --- mu(o).m(t) - m(0).mu(t); equal for ergodic systems
    R = _corr(val1, val2, cc_mode)

    # --- filtering
    # --- ToDo: another keyword for removing size dependent filter without
    # other filter (i.e. -0)
    if flt_pow < 0:
        # --- remove implicit size-dependent triangular filter (finite size)
        _filter = n_frames * np.ones(n_frames) - (np.arange(n_frames)-1)
        R /= _filter[:, None]

    if flt_pow != 0:
        _filter = signal_filter(n_frames, filter_type='welch') ** abs(flt_pow)
        R *= _filter[:, None]

    if not sum_dims:
        return R
    else:
        return R.sum(axis=1)


def spectral_density(*args, ts=1, factor=1/(2*np.pi), **kwargs):
    '''Calculate the spectral distribution as the Fourier transformed
       time-correlation function (TCF) of a vector signal (*args).
       The method automatically chooses to calculate auto- or cross-
       correlation functions based on the number of arguments (max 2).
       Adding signal filters may be enabled; use flt_pow=-1 to remove the
       implicit triangular filter due to finite size.
       Expects signal of shape (n_frames, n_dim)
       Keyword ts: timestep
       Returns:
        1 - discrete sample frequencies f = omega/(2*pi)
        2 - spectral density (FT TCF) as f(omega)
        3 - time-correlation function (timestep as in input)
       '''

    # --- enforce summation over dimensions
    kwargs.update({'sum_dims': True})
    R = time_correlation_function(*args, **kwargs)

    # --- \ --> \/
    # --- numpy convolve has given the full periodic cc function, but we cut
    #     it in half (see time_correlation_function()

    final_cc = np.hstack((R, R[::-1]))
    n = final_cc.shape[0]

    S = np.fft.rfft(final_cc, n=n).real * factor * ts

    # --- Prefactor: see Fourier Integral Theorem;
    #                Convention: factor = 1 / (2 pi), i.e.
    #                multiply by 2 pi where omega is actually put
    #                in place:
    #                EXAMPLE: \int_{-\infty}^{\infty} d\omega
    #                         dw = 2 pi * 1 / (dt * 2 * n_frames)
    #                         here: dt = ts = 1
    #                         numerical sum over omega corresponds to
    #                         \int_{0}^{\infty} d\omega --> factor 2 needed
    #                         (for symmetric/even integrand)
    #                         it follows: w_factor = dw * 2 = 2 pi / n_frames
    #
    #                In spectroscopy the prefactor is often used in
    #                forward FT with exp(-iwt).
    #                NB: 2 pi is a consequence of using omega and does not
    #                inherently stem from the Fourier transform

    # --- cycles per ts (NOT the angular frequency, NOT omega!)
    f = np.fft.rfftfreq(n, d=ts)
    # omega = 2 * np.pi * f

    return f, S, R
