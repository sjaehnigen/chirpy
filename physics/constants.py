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
import copy
from periodictable import elements as _EL
import warnings as _warnings

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

# --- A.U. conversion factors
l_au = a0_si  # Bohr radius [m]
E_au = 4.35974417E-18  # Hartree energy [J]
t_au = hbar_si / E_au  # time in A.U. [s]
m_p_au = m_p_si / m_e_si  # proton mass in A.U. [1] ?
m_amu_au = m_amu_si / m_e_si  # atomic mass unit in A.U. [1] ?
c_au = c_si / l_au * t_au  # speed of light in A.U. [1]
k_B_au = k_B_si / E_au  # Boltzmann constant [E_h/K]

# --- cgs units
c_cgs = c_si / centi  # speed of light [cm/s]
e_cgs = e_si * c_si * 1E1  # 4.80320427E-10 [esu]
h_cgs = h_si * kilo / centi**2  # Planck's constant [erg s]
hbar_cgs = h_cgs / (2*pi)  # reduced Planck constant [erg s]
a0_cgs = a0_si / centi  # Bohr radius [cm]
k_B_cgs = k_B_si * kilo / centi**2  # Boltzmann constant [erg/K]

# --- misc
a_lat = 2*pi / l_au  # lattice constant
finestr = 1 / c_au  # finestructure constant
l_au2aa = l_au * 1E10  # convertion A.U. to Angstrom
l_aa2au = 1 / l_au2aa  # convertion Angstrom to A.U.
t_au2fs = t_au / femto
t_fs2au = 1 / t_au2fs
v_au2si = 1E+5 * l_au2aa / t_au2fs
v_si2au = 1 / v_au2si
v_au2aaperfs = l_au2aa / t_au2fs

# --- spectroscopic light energy conversion functions
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


# --- other spectroscopic 
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


def _get_property(kinds, key):
    pr = []
    for _k in kinds:
        try:
            _r = getattr(elements[_k], key)
        except (KeyError, AttributeError):
            _warnings.warn('Cannot find %s for atom: %s !' % (key, _k),
                           RuntimeWarning, stacklevel=2)
            _r = None
        pr.append(_r)
    return pr


def numbers_to_symbols(numbers):
    return tuple(_get_property(numbers, 'symbol'))


def symbols_to_numbers(symbols):
    return _get_property(symbols, 'number')


def symbols_to_masses(symbols):
    return np.array(_get_property(symbols, 'mass'))


def symbols_to_valence_charges(symbols):
    return np.array(_get_property(symbols, 'valence_charge'))


def symbols_to_rvdw(symbols):
    return np.array(_get_property(symbols, 'van_der_waals_radius'))


numbers_to_masses = symbols_to_masses
numbers_to_valence_charges = symbols_to_valence_charges
numbers_to_rvdw = symbols_to_rvdw
