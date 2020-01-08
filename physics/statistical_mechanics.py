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
