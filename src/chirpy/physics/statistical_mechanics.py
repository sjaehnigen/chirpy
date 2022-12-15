# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
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


import numpy as np
import scipy
from .. import constants


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
    if filter_length > n_frames:
        filter_length = n_frames
    if filter_type == 'hanning':
        _window_func = np.hanning(2 * filter_length)[filter_length:]
    elif filter_type == 'welch':
        _window_func = (np.arange(filter_length)[::-1]/(filter_length+1))**2
    elif filter_type == 'triangular':
        _window_func = np.ones(filter_length) - \
                np.arange(filter_length)/filter_length
    else:
        raise ValueError('Filter %s not supported!' % filter_type)

    return np.pad(
            _window_func,
            (0, n_frames-filter_length),
            'constant',
            constant_values=(0, 0),
            )


def time_correlation_function(*args,
                              mode='full',
                              sum_dims=True,
                              finite_size_correction=True,
                              window_length=None,
                              flt_pow=-1.E-16,  # deprecated
                              ):
    '''Calculate the time-correlation function (TCF) of a signal and
       using the Wiener-Khinchin theorem (fftconvolve).
       The method automatically chooses to calculate auto- or cross-
       correlation functions based on the number of arguments (max 2).

       Using window functions (Welch) can be enabled through specifying
       the window_length argument.

       finite_size_correction ... Remove the implicit triangular filter
                                  due to finite size of the signal

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

    def _corr(_val1, _val2):
        _sig = np.array([scipy.signal.correlate(
                                                v1,
                                                v2,
                                                mode='full',
                                                method='fft',
                                                )
                         for v1, v2 in zip(_val1.T, _val2.T)]).T
        return _sig

    R = _corr(val1, val2)

    # --- filtering
    if flt_pow < 0 or finite_size_correction:
        # --- remove implicit size-dependent triangular filter (finite size)
        _filter = n_frames * np.ones(n_frames) - (np.arange(n_frames))
        _filter = np.hstack((_filter[:0:-1], _filter))
        R /= _filter[:, None]

    if flt_pow != 0 or window_length is not None:
        if window_length is None:
            _filter = signal_filter(n_frames,
                                    filter_type='welch') ** abs(flt_pow)
        else:
            _filter = signal_filter(n_frames,
                                    filter_type='welch',
                                    filter_length=window_length)

        _filter = np.hstack((_filter[::-1], _filter[1:]))
        R *= _filter[:, None]
        # R = (R - R[0]) * _filter[:, None] + R[0]

    # cc mode:
    # --- mu(o).m(t) - m(0).mu(t); equal for ergodic systems
    #   A or B = Ra or Rb
    #   AB = (Ra + Rb) / 2
    #   AC = Ra - Rb (benchmark)
    #   BC = 0
    #   ABC = A
    fR = np.zeros_like(val1)
    if 'A' in mode or 'C' in mode:
        fR += R[n_frames-1:]
    if 'B' in mode:
        fR += R[:n_frames][::-1]
    if 'C' in mode:
        fR -= R[:n_frames][::-1]
    if mode in ['AB', 'AC']:
        fR = fR / 2.
        # --- avoid double index 0 after vstack
        #     --> otherwise ugly phase shift in spectra
        #     --> not necessary for index -1 (cc = 0)
        # --- impose ergodicity / symmetry
        if mode == 'AB':
            fR = np.vstack((fR, fR[:0:-1]))
        if mode == 'AC':
            fR = np.vstack((fR, -fR[:0:-1]))

    if mode == 'full':
        fR = np.roll(R, len(R) // 2 + 1, axis=0)

    if not sum_dims:
        return fR
    else:
        return fR.sum(axis=1)


def spectral_density(*args, ts=1, factor=1/(2*np.pi), symmetry=None,
                     **kwargs) -> tuple:
    '''Calculate the spectral distribution as the Fourier transformed
       time-correlation function (TCF) of a vector signal (*args).
       The method automatically chooses to calculate auto- or cross-
       correlation functions based on the number of arguments (max 2).
       Adding signal filters may be enabled usign window_length=<float>
       finite_size_correction=True to remove implicit triangular filter due
       to finite size signal.
       Expects signal of shape (n_frames, n_dim)
       Keyword ts: timestep
       Signal symmetry can be even or odd (or None to get the full complex FT)
       Returns:
        1 - discrete sample frequencies f = omega/(2*pi)
        2 - spectral density (FT TCF) as f(omega)
        3 - time-correlation function (timestep as in input)
       '''

    # --- enforce summation over dimensions
    kwargs.update({'sum_dims': True})
    R = time_correlation_function(*args, **kwargs)

    if symmetry == 'even':
        n = R.shape[0]
        S = np.fft.rfft(R, n=n).real * factor * ts
    elif symmetry == 'odd':
        n = R.shape[0]
        # NB: minus sign
        S = -np.fft.rfft(R, n=n).imag * factor * ts
    elif symmetry is None:
        n = R.shape[0]
        S = np.fft.rfft(R, n=n) * factor * ts
    else:
        raise ValueError(f'unkown symmetry {symmetry}')

    # --- Prefactor: see Fourier Integral Theorem;
    #                Convention: factor = 1 / (2 pi), i.e.
    #                multiply by 2 pi where omega is actually put
    #                in place:
    #                EXAMPLE: \int_{-\infty}^{\infty} d\omega
    #                         dw = 2 pi * 1 / (dt * n)

    #                         if numerical sum over omega corresponds to
    #                         \int_{0}^{\infty} d\omega --> factor 2 needed
    #                         (for symmetric/even integrand)
    #                           w_factor = dw * 2 = 2 pi * 2 / n
    #
    #                         else if numerical sum over omega corresponds to
    #                         \int_{-\infty}^{\infty} d\omega
    #                           w_factor = dw = 2 pi / n
    #
    #                         NB: f[1] = 1 / (dt * n)
    #                         here: dt = ts = 1
    #
    #                In spectroscopy the prefactor is often used in
    #                forward FT with exp(-iwt).
    #                NB: 2 pi is a consequence of using omega and does not
    #                inherently stem from the Fourier transform

    # --- cycles per ts (NOT the angular frequency, NOT omega!)
    f = np.fft.rfftfreq(n, d=ts)
    # omega = 2 * np.pi * f

    return f, S, R
