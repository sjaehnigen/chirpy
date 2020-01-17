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
from scipy import signal
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


def maxwell_boltzmann_distribution(T_K, *args, **kwargs):
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
        return _options.get(kwargs.get('option', 'energy'))(T_K, x, *args)

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


def spectral_density(*args, **kwargs):
    '''Calculate the spectral distribution of a signal (*aurgs) over frequency,
       based on the Fourier transformed time-correlation function (TCF) of that
       signal and the Wiener-Khinchin theorem (fftconvolve).
       The method automatically chooses to calculate auto- or cross-
       correlation functions based on the number of arguments (max 2).
       Adding signal filters may be enabled; use flt_pow=-1 to remove the
       implicit triangular filter due to finite size.
       Returns:
        1 - discrete sample frequencies
        2 - spectral density (FT TCF)
        3 - time-correlation function (timestep as in input)
       '''

    if len(args) == 1:
        auto = True
        val1 = args[0]

    elif len(args) == 2:
        auto = False
        val1 = args[0]
        val2 = args[1]

    else:
        raise TypeError('spectral_density takes at most 2 arguments, got %d'
                        % len(args))

    ts = kwargs.get('ts', 4)
    flt_pow = kwargs.get('flt_pow', 0)
    _fac = kwargs.get('factor', 1)
    _cc_mode = kwargs.get('cc_mode', 'AB')

    n_frames, three = val1.shape

    def _corr(_val1, _val2, mode='full'):
        return np.array([signal.fftconvolve(
                                  v1,
                                  v2[::-1],
                                  mode=mode
                                  )[n_frames-1:]
                         for v1, v2 in zip(_val1.T, _val2.T)]).T

    if auto:
        R = _corr(val1, val1)

    else:
        R = np.zeros_like(val1)
        # cc mode:
        #   A or B = Ra or Rb
        #   AB = (Ra + Rb) / 2
        #   AC = Ra - Rb (benchmark)
        #   BC = 0
        #   ABC = A
        # --- mu(o).m(t) - m(0).mu(t); equal for ergodic systems

        if 'A' in _cc_mode:
            R += _corr(val1, val2)

        if 'B' in _cc_mode:
            R += _corr(val2, val1)
            # R += np.array([signal.fftconvolve(
            #                v2, v1[::-1], mode='full')[n_frames-1:]
            #                for v1, v2 in zip(val1.T, val2.T)]).T

        if 'C' in _cc_mode:
            R -= _corr(val2, val1)

        if _cc_mode == 'AB':
            R /= 2

    R = R.sum(axis=1)

    # --- filtering
    if flt_pow > 0:
        _filter = signal_filter(n_frames, filter_type='welch') ** flt_pow
        R *= _filter

    elif flt_pow == -1:
        # --- remove implicit size-dependent triangular filter
        R /= (n_frames * np.ones(n_frames) - (np.arange(n_frames)-1))

    # --- \ --> /\
    final_cc = np.hstack((R, R[::-1]))
    n = final_cc.shape[0]

    S = np.fft.rfft(final_cc, n=n-1).real * ts * constants.t_fs2au * _fac
    # S /= 2*np.pi
    omega = np.fft.rfftfreq(n-1, d=ts)

    return omega, S, R
