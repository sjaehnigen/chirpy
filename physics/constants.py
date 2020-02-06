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


import numpy as np
import warnings as _warnings

# metric prefixes
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

# misc constants
pi = np.pi  # pi
avog = 6.02214129E+23  # Avogadro constant [mol-1]

# S.I. constants
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

# A.U. conversion factors
l_au = a0_si  # Bohr radius [m]
E_au = 4.35974417E-18  # Hartree energy [J]
t_au = hbar_si / E_au  # time in A.U. [s]
m_p_au = m_p_si / m_e_si  # proton mass in A.U. [1] ?
m_amu_au = m_amu_si / m_e_si  # atomic mass unit in A.U. [1] ?
c_au = c_si / l_au * t_au  # speed of light in A.U. [1]
k_B_au = k_B_si / E_au  # Boltzmann constant [E_h/K]

# cgs units
c_cgs = c_si / centi  # speed of light [cm/s]
e_cgs = e_si * c_si * 1E1  # 4.80320427E-10 [esu]
h_cgs = h_si * kilo / centi**2  # Planck's constant [erg s]
hbar_cgs = h_cgs / (2*pi)  # reduced Planck constant [erg s]
a0_cgs = a0_si / centi  # Bohr radius [cm]
k_B_cgs = k_B_si * kilo / centi**2  # Boltzmann constant [erg/K]

# misc
a_lat = 2*pi / l_au  # lattice constant
finestr = 1 / c_au  # finestructure constant
l_au2aa = l_au * 1E10  # convertion A.U. to Angstrom
l_aa2au = 1 / l_au2aa  # convertion Angstrom to A.U.
t_au2fs = t_au / femto
t_fs2au = 1 / t_au2fs
v_au2si = 1E+5 * l_au2aa / t_au2fs
v_si2au = 1 / v_au2si
v_au2aaperfs = l_au2aa / t_au2fs

# spectroscopic light energy conversion functions
E_eV2J = e_si
E_J2eV = 1 / e_si
E_Hz2J = h_si
E_J2Hz = 1 / E_Hz2J
E_Hz2cm_1 = 1 / c_si * centi
E_J2cm_1 = E_J2Hz * E_Hz2cm_1
E_eV2cm_1 = E_eV2J * E_J2cm_1


def E_J2nm(x):
    return h_si * c_si / x / nano


def E_nm2J(x):
    return h_si * c_si / x * nano


def E_Hz2nm(x):
    return E_J2nm(x * E_Hz2J)


# Comments for all non-PSE element symbols mandatory
elements = np.array([
    ('H',   1.00797,  1, 1, 110.0, ''),
    ('D',   2.01410,  1, 1, 110.0, 'isotope'),
    ('He',  4.00260,  2, 2, 140.0, ''),
    ('Li',  6.93900,  3, 3, 182.0, ''),
    ('Be',  9.01220,  4, 4, 153.0, ''),
    ('B',   0.0,      5, 0,   1.0, ''),
    ('C',  12.01115,  6, 4, 170.0, ''),
    ('N',  14.00670,  7, 5, 155.0, ''),
    ('O',  15.99940,  8, 6, 152.0, ''),
    ('F',  18.99840,  9, 7, 147.0, ''),
    ('Ne', 20.17976, 10, 0, 154.0, ''),
    ('Na', 22.98980, 11, 1, 227.0, ''),
    ('Mg',  0.0,     12, 2,   1.0, ''),
    ('Al',  0.0,     13, 0,   1.0, ''),
    ('Si', 28.085,   14, 4, 210.0, ''),
    ('P',  30.97376, 15, 5, 180.0, ''),
    ('S',  32.06,    16, 6, 180.0, ''),
    ('Cl', 35.45,    17, 7, 175.0, ''),
    ('Ar', 39.95,    18, 0, 188.0, ''),
    # ('K', , , , ),
    # ('Ca', 39.96259, 20, 2, 231.0),  # Unverified
    # ('Sc', , , ,),
    # ('Ti', , , ,),
    # ('V', , , ,),
    # ('Cr', , , ,),
    # ('Mn', , , ,),
    # ('Fe', , , ,),
    # ('Co', , , ,),
    # ('Ni', , , ,),
    # ('Cu', , , ,),
    # ('Zn', , , ,),
    # ...
    ('X',   0.00000, 0, 0, 0.0, 'dummy'),
       ], dtype=[
           ('symbol', '<U2'),
           ('mass_amu', '<f8'),
           ('atomic_number', '<i8'),
           ('valence_charge', '<i8'),
           ('van_der_waals_radius', '<f8'),
           ('comment', '<U20'),
           ]
       ).view(np.recarray)


def _get_property_dict(key):
    return {_s: _m for _s, _m in zip(elements.symbol, getattr(elements, key))}


def numbers_to_symbols(numbers):
    return tuple(
            [elements.symbol[elements.comment == ''][_n-1] for _n in numbers]
            )


def numbers_to_masses(numbers):
    '''No isotope support (use symbols_to_masses)'''
    return [elements.mass_amu[elements.comment == ''][_n-1] for _n in numbers]


def symbols_to_numbers(symbols):
    try:
        return [atomic_numbers[_z.title()] for _z in symbols]

    except KeyError:
        _warnings.warn('Could not find all elements! '
                       'Numbers cannot be used.',
                       RuntimeWarning, stacklevel=2)

    except AttributeError:
        _warnings.warn('Got wrong format for symbols! '
                       'Numbers cannot be used.',
                       RuntimeWarning, stacklevel=2)


def symbols_to_masses(symbols):
    try:
        return np.array([masses_amu[_z.title()] for _z in symbols])

    except KeyError:
        _warnings.warn('Could not find masses for all elements! '
                       'Centre of mass cannot be used.',
                       RuntimeWarning, stacklevel=2)

    except AttributeError:
        _warnings.warn('Got wrong format for symbols! '
                       'Centre of mass cannot be used.',
                       RuntimeWarning, stacklevel=2)


def symbols_to_valence_charges(symbols):
    try:
        return np.array([valence_charges[_z.title()] for _z in symbols])

    except KeyError:
        _warnings.warn('Could not find all elements! '
                       'Numbers cannot be used.',
                       RuntimeWarning, stacklevel=2)

    except AttributeError:
        _warnings.warn('Got wrong format for symbols! '
                       'Numbers cannot be used.',
                       RuntimeWarning, stacklevel=2)


def symbols_to_rvdw(symbols):
    return np.array([rvdw[_z.title()] for _z in symbols])


atomic_numbers = _get_property_dict("atomic_number")
masses_amu = _get_property_dict("mass_amu")
valence_charges = _get_property_dict("valence_charge")
rvdw = _get_property_dict("van_der_waals_radius")


eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


IR_au2kmpmol = (avog * e_si**2) / (12 * eps0_si * c_si**2 * m_amu_si * kilo)


def _dipole_dipole_prefactor(T_K):
    beta_cgs = 1./(T_K * k_B_cgs)
    prefactor_cgs = (2 * np.pi * avog * beta_cgs * finestr * hbar_cgs) / 3
    return prefactor_cgs


def current_current_prefactor(T_K):
    prefactor_cgs = _dipole_dipole_prefactor(T_K) * (a0_cgs/t_au)**2
    cm2_kmcm = 1E-5
    return prefactor_cgs * cm2_kmcm * t_au


def current_magnetic_prefactor(nu_cgs, T_K):
    prefactor_cgs = _dipole_dipole_prefactor(T_K) * (a0_cgs**3/t_au**2/c_cgs)
    cm2_kmcm = 1E-5
    omega_cgs = nu_cgs * c_cgs * 2 * np.pi
    return 4 * omega_cgs * prefactor_cgs * cm2_kmcm * t_au
