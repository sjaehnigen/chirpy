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

# misc constants
pi = np.pi  # pi
avog = 6.02214129E+23  # Avogadro constant [mol-1]

# S.I. constants
c_si = 2.99792458E+08  # speed of light [m/s]
h_si = 6.62606957E-34  # Planck's constant [Js]
hbar_si = h_si/(2*pi)  # reduced planck constant [Js]
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
t_au = hbar_si/E_au  # time in A.U. [s]
m_p_au = m_p_si/m_e_si  # proton mass in A.U. [1] ?
m_amu_au = m_amu_si/m_e_si  # atomic mass unit in A.U. [1] ?
c_au = c_si/l_au*t_au  # speed of light in A.U. [1]
k_B_au = k_B_si/E_au  # Boltzmann constant [E_h/K]

# cgs units
c_cgs = c_si*1E2  # speed of light [cm/s]
e_cgs = e_si*c_si*1E1  # 4.80320427E-10 [esu]
h_cgs = h_si*1E7  # Planck's constant [erg s]
hbar_cgs = h_cgs/(2*pi)  # reduced Planck constant [erg s]
a0_cgs = a0_si*1E2  # Bohr radius [cm]
k_B_cgs = k_B_si*1E7  # Boltzmann constant [erg/K]

# misc
a_lat = 2*pi/l_au  # lattice constant
finestr = 1/c_au  # finestructure constant
l_au2aa = l_au*1E10  # convertion A.U. to Angstrom: x(in au)*l_au2aa = x(in aa)
l_aa2au = 1/l_au2aa  # convertion Angstrom to A.U.: x(in aa)*l_aa2au = x(in au)
t_au2fs = t_au*1E15
t_fs2au = 1/t_au2fs
v_au2si = 1E+5*l_au2aa/t_au2fs
v_si2au = 1/v_au2si
v_au2aaperfs = l_au2aa/t_au2fs

# element species
# species = dict()
# species['H']  = {'SYMBOL':  'H', 'MASS':  1.00797, 'Z':  1, 'ZV':  1, 'RVDW': 110.0}
# species['D']  = {'SYMBOL':  'H', 'MASS':  2.01410, 'Z':  1, 'ZV':  1, 'RVDW': 110.0}
# species['He'] = {'SYMBOL': 'He', 'MASS':  4.00260, 'Z':  2, 'ZV':  2, 'RVDW': 140.0}
# species['Li'] = {'SYMBOL': 'Li', 'MASS':  6.93900, 'Z':  3, 'ZV':  3, 'RVDW': 182.0}
# species['Be'] = {'SYMBOL': 'Be', 'MASS':  9.01220, 'Z':  4, 'ZV':  4, 'RVDW': 153.0}
# species['C']  = {'SYMBOL':  'C', 'MASS': 12.01115, 'Z':  6, 'ZV':  4, 'RVDW': 170.0}
# species['N']  = {'SYMBOL':  'N', 'MASS': 14.00670, 'Z':  7, 'ZV':  5, 'RVDW': 155.0}
# species['O']  = {'SYMBOL':  'O', 'MASS': 15.99940, 'Z':  8, 'ZV':  6, 'RVDW': 152.0}
# species['F']  = {'SYMBOL':  'F', 'MASS': 18.99840, 'Z':  9, 'ZV':  7, 'RVDW': 147.0}
# species['Ne'] = {'SYMBOL': 'Ne', 'MASS': 20.18300, 'Z': 10, 'ZV':  0, 'RVDW': 154.0}
# species['Na'] = {'SYMBOL': 'Na', 'MASS': 22.98980, 'Z': 11, 'ZV':  0, 'RVDW': 227.0}
# species['Si'] = {'SYMBOL': 'Si', 'MASS': 28.085  , 'Z': 14, 'ZV':  4, 'RVDW': 210.0}
# species['P']  = {'SYMBOL': 'P',  'MASS': 30.97376, 'Z': 15, 'ZV':  5, 'RVDW': 180.0}
# species['S']  = {'SYMBOL': 'S',  'MASS': 32.06400, 'Z': 16, 'ZV':  6, 'RVDW': 180.0}
# species['Cl'] = {'SYMBOL': 'Cl', 'MASS': 35.45300, 'Z': 17, 'ZV':  0, 'RVDW': 175.0}
# species['Ca'] = {'SYMBOL': 'Ca', 'MASS': 39.96259, 'Z': 20, 'ZV':  0, 'RVDW': 231.0}
# species['X']  = {'SYMBOL':  'X', 'MASS':  0.00000, 'Z':  0, 'ZV':  0, 'RVDW':   0.0}


# Comments for all non-PSE element symbols mandatory
elements = np.array([
    ('H',   1.00797,  1, 1, 110.0, ''),
    ('D',   2.01410,  1, 1, 110.0, 'isotope'),
    ('He',  4.00260,  2, 2, 140.0, ''),
    ('Li',  6.93900,  3, 3, 182.0, ''),
    ('Be',  9.01220,  4, 4, 153.0, ''),
    ('B',   0.0,      5, 0,   0.0, ''),
    ('C',  12.01115,  6, 4, 170.0, ''),
    ('N',  14.00670,  7, 5, 155.0, ''),
    ('O',  15.99940,  8, 6, 152.0, ''),
    ('F',  18.99840,  9, 7, 147.0, ''),
    ('Ne', 20.17976, 10, 0, 154.0, ''),
    ('Na', 22.98980, 11, 1, 227.0, ''),
    ('Mg',  0.0,     12, 2,   0.0, ''),
    ('Al',  0.0,     13, 0,   0.0, ''),
    ('Si', 28.085,   14, 4, 210.0, ''),
    ('P',  30.97376, 15, 5, 180.0, ''),
    ('S',  32.06,    16, 6, 180.0, ''),
    ('Cl', 35.45,    17, 7, 175.0, ''),
    ('Ar', 39.95,    18, 0, 188.0, ''),
    # K
    # ('Ca', 39.96259, 20, 2, 231.0),  # Unchecked
    # Sc
    # Ti
    # V
    # Cr
    # Mn
    # Fe
    # Co
    # Ni
    # Cu
    # Zn
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
    return [elements.symbol[elements.comment == ''][_n-1] for _n in numbers]


def numbers_to_masses(numbers):
    '''No isotope support (use symbols_to_masses)'''
    return [elements.mass_amu[elements.comment == ''][_n-1] for _n in numbers]


def symbols_to_numbers(symbols):
    return [atomic_numbers[_z.title()] for _z in symbols]


def symbols_to_masses(symbols):
    return [masses_amu[_z.title()] for _z in symbols]


def symbols_to_rvdw(symbols):
    return [rvdw[_z.title()] for _z in symbols]


atomic_numbers = _get_property_dict("atomic_number")
masses_amu = _get_property_dict("mass_amu")
valence_charges = _get_property_dict("valence_charge")
rvdw = _get_property_dict("van_der_waals_radius")

# Get rid of this list
# element symbols
# don't add Deuterium to this list since cubefiletools identifies species by atomic number
# symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'X']

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
