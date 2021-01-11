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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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


import warnings as _warnings
# --- ChirPy uses "CP2K convention"

_warnings.warn('Mandatory units are specified by variable suffix _<unit> '
               '(e.g., pos_aa)', stacklevel=2)
_warnings.warn('au = atomic units', stacklevel=2)
_warnings.warn('aa = ångström', stacklevel=2)
_warnings.warn('amu = atomic mass unit', stacklevel=2)
_warnings.warn('Modules in physics expect atomic units for all variables. '
               'Make sure to convert input before calling the methods!',
               stacklevel=2)
_warnings.warn('Derived units follow the convention rigorously.',
               stacklevel=2)

# --- atomic units
velocity = 'au'
energy = 'au'
magnetic_dipole = ('au', 'no-cgs')
current_dipole = 'au'
current = 'au'
electric_field = 'au'
magnetic_field = ('au', 'no-cgs')
hessian = 'au'
vibrational_mode = 'au'
frequency = 'au'

# --- other
angle = 'rad'
cell = 'aa_deg'
time = 'fs'
temperature = 'K'
length = 'aa'
mass = 'amu'
ir_intensity = 'kmpmol'
rotional_strength = 'kmpmol'
