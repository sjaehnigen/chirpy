# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
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
# -------------------------------------------------------------------


import numpy as np
import copy
from periodictable import elements as _EL
import warnings as _warnings
from . import config

# --- Levi-Civita
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

# --- metric prefixes
peta = 1.E+15
tera = 1.E+12
giga = 1.E+9
mega = 1.E+6
kilo = 1.E+3
dezi = 1.E-1
centi = 1.E-2
milli = 1.E-3
micro = 1.E-6
nano = 1.E-9
pico = 1.E-12
femto = 1.E-15

# --- misc constants
pi = np.pi  # pi
avog = 6.02214129E+23  # Avogadro constant [mol-1]

# --- S.I. constants
c_si = 2.99792458E+08  # speed of light [m/s]
h_si = 6.62606957E-34  # Planck's constant [Js]
hbar_si = h_si / (2*pi)  # reduced planck constant [Js]
m_p_si = 1.67262177E-27  # mass protron [kg]
m_e_si = 9.10938291E-31  # mass electron [kg]
m_amu_si = 1.660538921E-27  # atomic mass unit [kg]
e_si = 1.60217657E-19  # unit charge [C]
eps0_si = 8.854187817E-12  # vacuum permittivity
a0_si = 0.529177210859E-10  # Bohr radius [m]
k_B_si = 1.3806488E-23  # Boltzmann constant [J/K]
mu_b_si = e_si * hbar_si / 2 / m_e_si  # Bohr magneton [J/T]

# --- A.U. conversion factors
l_au = a0_si  # Bohr radius [m]
E_au = 4.35974417E-18  # Hartree energy [J]
t_au = hbar_si / E_au  # time in A.U. [s]
m_p_au = m_p_si / m_e_si  # proton mass in A.U. [1] ?
m_amu_au = m_amu_si / m_e_si  # atomic mass unit in A.U. [1] ?
c_au = c_si / l_au * t_au  # speed of light in A.U. [1]
k_B_au = k_B_si / E_au  # Boltzmann constant [E_h/K]
mu_b_au = 0.5  # Bohr magneton

# --- cgs units
c_cgs = c_si / centi  # speed of light [cm/s]
e_cgs = e_si * c_si * 1E1  # 4.80320427E-10 [esu]
h_cgs = h_si * kilo / centi**2  # Planck's constant [erg s]
hbar_cgs = h_cgs / (2*pi)  # reduced Planck constant [erg s]
a0_cgs = a0_si / centi  # Bohr radius [cm]
k_B_cgs = k_B_si * kilo / centi**2  # Boltzmann constant [erg/K]
# --- esu extension (= Gaussian-cgs for electric units)
q_si2esu = c_si * 1E1  # charge, statC, Franklin
q_esu2si = 1 / q_si2esu
p_debye2si = 1. / c_si * 1E-21  # electric dipole moment
p_si2debye = 1. / p_debye2si
p_debye2au = p_debye2si / e_si / a0_si
p_au2debye = 1 / p_debye2au
# --- emu extension (= Gaussian-cgs for magnetic units)
mu_b_emu = e_cgs * hbar_cgs / 2 / c_cgs / m_e_si / 1E3  # Bohr magneton [erg/G]


# --- misc
a_lat = 2*pi / l_au  # lattice constant
finestr = 1 / c_au  # finestructure constant
l_au2aa = l_au * 1E10  # convertion A.U. to Angstrom
l_aa2au = 1 / l_au2aa  # convertion Angstrom to A.U.
t_au2fs = t_au / femto
t_fs2au = 1 / t_au2fs
v_au2si = 1E+5 * l_au2aa / t_au2fs
v_si2au = 1 / v_au2si
v_au2aa_fs = l_au2aa / t_au2fs
p_si2au = 1 / e_si / a0_si
p_au2si = 1 / p_si2au

# --- spectroscopic light energy and frequency conversion functions
E_au2J = E_au
E_J2au = 1 / E_au
E_eV2J = e_si
E_J2eV = 1 / e_si
E_Hz2J = h_si
E_J2Hz = 1 / E_Hz2J
E_Hz2cm_1 = centi / c_si
E_cm_12Hz = 1 / E_Hz2cm_1
E_J2cm_1 = E_J2Hz * E_Hz2cm_1
E_cm_12J = 1 / E_J2cm_1
E_eV2cm_1 = E_eV2J * E_J2cm_1
E_cm_12eV = E_cm_12J * E_J2eV
E_eV2Hz = E_eV2J * E_J2Hz
E_Hz2eV = E_Hz2J * E_J2eV
# --- NB: atomic units of 1/time and not Hartree energy!
E_cm_12aufreq = E_cm_12Hz * t_au
E_aufreq2cm_1 = 1 / t_au * E_Hz2cm_1
E_Hz2aufreq = t_au
E_aufreq2Hz = 1 / t_au


def E_J2nm(x):
    return h_si * c_si / x / nano


def E_nm2J(x):
    return h_si * c_si / x * nano


def E_Hz2nm(x):
    return E_J2nm(x * E_Hz2J)


# --- other spectroscopic
#     Calculated spectra are given as specific absorption coefficient according
#     to Beer-Lambert law with general unit 1/(amount_per_volume * distance)
#     (for continuous or Lorentzian-broadened spectra and TCF calculations)
Abs_au2si_mol = avog * l_au**2
Abs_au2L_cm_mol = Abs_au2si_mol * centi / dezi**3

#     Integrated absorption coefficient with general units
#     frequency-based:   1/(amount_per_volume * distance * time) or
#     wavenumber-based:  distance/amount
#     (for line spectra and static calculations)
IntAbs_au2si_mol = Abs_au2si_mol / t_au
IntAbs_au2km_mol = IntAbs_au2si_mol / c_si / kilo


# --- ToDo: backward compatibility
Abs_au2L_per_cm_mol = Abs_au2L_cm_mol
Abs_au2si_per_mol = Abs_au2si_mol
IntAbs_au2si_per_mol = IntAbs_au2si_mol
IntAbs_au2km_per_mol = IntAbs_au2km_mol


def current_current_prefactor_au(T_K, n=1):
    '''in time / charge**2'''
    # --- from Fermi's Golden Rule we have factor of omega
    # --- finestr equals e**2 / (4 pi eps_0) / (hbar * c)
    # --- we multiply with omega * hbar * beta (classical limit for Kubo TCF)
    # --- transition to current dipole moment squared gives factor 1/omega**2
    # --- see also McQuarrie, Statisical Mechanics, Appendix F
    beta_au = 1 / (T_K * k_B_au)
    prefactor_au = 4 * np.pi**2 * finestr / 3 / n * beta_au
    return prefactor_au


def dipole_dipole_prefactor_au(T_K, omega_au, n=1):
    '''in 1 / (time * charge**2)
       omega_au = 2 * pi * freq_au'''
    prefactor_au = current_current_prefactor_au(T_K, n=n) * omega_au**2
    return prefactor_au


def current_magnetic_prefactor_au(T_K, omega_au, n=1):
    '''in time / (distance * charge**2)
       omega_au = 2 * pi * freq_au
       No cgs-convention for magnetic properties, i.e. unit of m is
       current * distance**2.
    '''
    # --- factor 1/c here because we do not use cgs for B-field

    prefactor_au = 4 * current_current_prefactor_au(T_K, n=n) * omega_au / c_au
    return prefactor_au


def dipole_magnetic_prefactor_au(T_K, omega_au, n=1):
    '''in 1 / (distance * charge**2)
       omega_au = 2 * pi * freq_au
       No cgs-convention for magnetic properties, i.e. unit of m is
       current * distance**2.
       '''
    # --- factor 1/c here because we do not use cgs for B-field

    prefactor_au = current_magnetic_prefactor_au(T_K, omega_au, n=n) * omega_au
    return prefactor_au


# PERIODIC TABLE
# --- load module representation from periodictable (mutable type!)
elements = _EL._element

# --- add symbol strings as keys to dict
elements.update({elements[_a].symbol: elements[_a] for _a in elements})
# --- add deuterium
elements['D'] = _EL.D
# --- add ghost atom (and overwrite entry for neutron in periodictable)
X = copy.deepcopy(_EL.n)
X.name = 'dummy'
X.symbol = 'X'
X.number = '0'
X._isotopes = {}
X.ions = (0)
X._mass = 0.0
X._density = 0.0
X.density_caveat = ''
elements[0] = X
elements['X'] = X
elements['0'] = X
# --- van der Waals radii in pm for some elements
_rvdw_list = [
    ('H',  110.0),
    ('D',  110.0),
    ('He', 140.0),
    ('Li', 182.0),
    ('Be', 153.0),
    # B
    ('C',  170.0),
    ('N',  155.0),
    ('O',  152.0),
    ('F',  147.0),
    ('Ne', 154.0),
    ('Na', 227.0),
    # Mg, Al
    ('Si', 210.0),
    ('P',  180.0),
    ('S',  180.0),
    ('Cl', 175.0),
    ('Ar', 188.0),
    # K
    ('Ca', 231.0),
    ('I',  198.0),
    ('X',    0.0),
    ]
for _z, _rvdw in _rvdw_list:
    elements[_z].van_der_waals_radius = _rvdw
    elements[_z].van_der_waals_radius_units = 'pm'
# --- valence charges (pseudopotentials) for some elements
_ZV_list = [
    ('H',  1),
    ('D',  1),
    ('He', 2),
    ('Li', 3),
    ('Be', 4),
    # B
    ('C',  4),
    ('N',  5),
    ('O',  6),
    ('F',  7),
    ('Ne', 0),
    # Na, Mg, Al
    ('Si', 4),
    ('P',  5),
    ('S',  6),
    ('Cl', 7),
    ('Ar', 0),
    # K, Ca
    ('X',  0),
    ]
for _z, _ZV in _ZV_list:
    elements[_z].valence_charge = _ZV


def _get_property(kinds, key, fmt=None, fill_value=None):
    pr = []
    for _k in kinds:
        _guess = _k
        while True:
            try:
                _r = getattr(elements[_guess], key)
                if _guess != _k:
                    _warnings.warn(f'Guessing element: {_k} --> {_guess}',
                                   config.ChirPyWarning,
                                   stacklevel=3)
                break

            except (KeyError, AttributeError):
                try:
                    _guess = _guess[:-1]
                    if len(_guess) == 0:
                        raise IndexError
                    continue

                except (IndexError, TypeError):
                    if fill_value == 'self':
                        _r = _k
                    else:
                        _r = fill_value
                    break

        if fmt is not None:
            pr.append(fmt(_r))
        else:
            pr.append(_r)
    # if config.__verbose__:
    #     _warnings.warn(f'Got {key} of {kinds}: {pr}',
    #                    config.ChirPyWarning, stacklevel=3)
    if None in pr:
        [_warnings.warn(f'Could not find {key} of \'{kinds[_ipr]}\' '
                        f'(id {_ipr}).',
                        config.ChirPyWarning,
                        stacklevel=4)
         for _ipr, _pr in enumerate(pr) if _pr is None]

    return pr


def numbers_to_symbols(numbers):
    return tuple(_get_property(numbers, 'symbol'))


def symbols_to_symbols(numbers):
    return tuple(_get_property(numbers, 'symbol', fill_value='self'))


def symbols_to_numbers(symbols):
    return _get_property(symbols, 'number', fmt=int)


def symbols_to_masses(symbols):
    return np.array(_get_property(symbols, 'mass'))


def symbols_to_valence_charges(symbols):
    return np.array(_get_property(symbols, 'valence_charge'))


def symbols_to_rvdw(symbols):
    return np.array(_get_property(symbols, 'van_der_waals_radius'))


numbers_to_masses = symbols_to_masses
numbers_to_valence_charges = symbols_to_valence_charges
numbers_to_rvdw = symbols_to_rvdw


def get_conversion_factor(name, unit):
    '''Return factor to convert given unit into default unit according to
       chirpy.version.<name>.
       name ... type of magnitude (positions, velocities, etc.)
       '''
    _db = {
            'length': {
                'aa': 1.,
                'au': l_au2aa,
                'si': 1E10,
                },
            'velocity': {
                'au': 1.,
                'aa': l_aa2au,  # time in a.u.
                'aa_fs': 1/v_au2aa_fs,
                'aa_ps': 1/v_au2aa_fs/1000,
                'si': v_si2au
                },
            'electric_dipole': {
                'au': 1.,
                'si': p_si2au,
                'debye': p_debye2au,
                'eaa': l_aa2au,  # electron-Ang
                },
            'current_dipole': {
                'au': 1.,
                # 'si': p_si2au,
                'debye_ps': p_debye2au/t_fs2au/1000,
                },
            'magnetic_dipole': {
                'au': 1.,
                'debyeaa_ps': p_debye2au/t_fs2au/1000 * l_aa2au,
                },
            # 'moment_v': {  # "velocity form"
            #     'au': 1.
            #     },
            }
    try:
        return _db[name][unit]
    except KeyError:
        if name in _db:
            raise ValueError(f'Could not find unit \'{unit}\' for \'{name}\'.')
        else:
            raise ValueError(f'Unknown magnitude \'{name}\' in units.')
        return 1.


def convert(units):
    if units == 'default':
        return 1.

    if isinstance(units, list):
        convert = np.array([get_conversion_factor(_i, _j)
                            for _i, _j in units])
    elif isinstance(units, tuple):
        convert = get_conversion_factor(*units)
    elif isinstance(units, (float, int)):
        convert = float(units)
    else:
        raise ValueError('invalid units')
    return convert
