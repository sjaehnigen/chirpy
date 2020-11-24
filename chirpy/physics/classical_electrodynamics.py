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


import numpy as np
from ..physics import constants, kspace
from ..physics.constants import eijk
from ..topology import mapping
from ..mathematics.analysis import divrot

# m magnetic dipole moment
# c current ipole moment


def electric_dipole_moment(pos_au, charges_au):
    return pos_au * charges_au[:, None]


def current_dipole_moment(vel_au, charges_au):
    return vel_au * charges_au[:, None]


def electric_quadrupole_moment(pos_au, charges_au):
    '''traceless'''

    return np.sum(
              (
               3 * np.einsum('mi, mj -> mij', pos_au, pos_au)
               - (pos_au**2).sum(axis=-1)[:, None, None] * np.identity(3)[None]
               ) * charges_au[:, None, None],
              axis=0
                  ) / 2


def electric_dipole_shift_origin(charges_au, trans_au):
    '''Compute differential term of origin shift
       charges_au ... atomic charges
       trans_au ... translation vector from old to new origin
       '''
    if len(charges_au.shape) == 1:
        return -trans_au * charges_au[None]

    if len(charges_au.shape) == 2:
        return -trans_au * charges_au[:, None]

    if len(charges_au.shape) == 3:
        return -trans_au * charges_au[:, :, None]


def magnetic_dipole_shift_origin(c_au, trans_au):
    '''Compute differential term of origin shift
       c_au ... current dipole moment
       trans_au ... translation vector from old to new origin
       NB: No cgs-convention
       '''
    if len(c_au.shape) == 1:
        return -0.5 * np.sum(eijk[:, :, :]
                             * trans_au[:, None, None]
                             * c_au[None, :, None],
                             axis=(0, 1))
    if len(c_au.shape) == 2:
        return -0.5 * np.sum(eijk[None, :, :, :]
                             * trans_au[:, :, None, None]
                             * c_au[:, None, :, None],
                             axis=(1, 2))
    if len(c_au.shape) == 3:
        return -0.5 * np.sum(eijk[None, None, :, :, :]
                             * trans_au[:, :, :, None, None]
                             * c_au[:, :, None, :, None],
                             axis=(2, 3))  # sum over mols (axis 1) done later


def switch_electric_origin_gauge(mu_au, charges_au, o_a_au, o_b_au,
                                 cell_au_deg=None):
    '''Apply (distrubuted) origin gauge on electric dipole moments shifting
       from origin A to origin B.
       Accepts cell_au_deg argument to account for periodic boundaries.
       Expects atomic units (no cgs-convention).

       mu_au      ... electric dipole moment of shape (N, 3) or (3)
       charges_au ... charges of shape (N) or float
       with N being the number of kinds/atoms/states.

       o_a_au ... old origin(s) of shape (N, 3) or (3)
       o_b_au ... new origin(s) of shape (N, 3) or (3)

       Returns: An updated array of mu_au
       '''

    _trans = mapping.distance_pbc(o_a_au, o_b_au, cell=cell_au_deg)

    return mu_au + electric_dipole_shift_origin(charges_au, _trans)


def switch_magnetic_origin_gauge(c_au, m_au, o_a_au, o_b_au, cell_au_deg=None):
    '''Apply (distrubuted) origin gauge on magnetic dipole moments shifting
       from origin A to origin B.
       Accepts cell_au_deg argument to account for periodic boundaries.
       Expects atomic units (no cgs-convention).

       c_au ... current dipole moment of shape (N, 3) or (3)
       m_au ... magnetic dipole moment (before the gauge transformation)
       Both of shape (N, 3) with N being the number of kinds/atoms/states.

       o_a_au ... old origin(s) of shape (N, 3) or (3)
       o_b_au ... new origin(s) of shape (N, 3) or (3)

       Returns: An updated array of m_au
       '''

    _trans = mapping.distance_pbc(o_a_au, o_b_au, cell=cell_au_deg)

    return m_au + magnetic_dipole_shift_origin(c_au, _trans)


def coulomb(r0, r, q, cell=None, thresh=1.E-8):
    '''r...shape(N, ..., 3)'''
    d = mapping.distance_pbc(r, r0, cell=cell)  # r0 - r
    d3 = np.linalg.norm(d, axis=-1)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3 < thresh**3, 0.0, np.divide(1.0, d3))
    E = q * d * d3_inv[:, None]
    return E


def coulomb_grid(r, rho, pos_grid, voxel, cell=None, thresh=1.E-8):
    '''r...shape(3, ..., N)'''
    if cell is not None:
        raise NotImplementedError('coulomb_grid does not support periodic '
                                  'boundaries!')
    d = r[:, None, None, None]-pos_grid
    d3 = np.linalg.norm(d, axis=0)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3 < thresh**3, 0.0, np.divide(1.0, d3))
    E = rho[None] * d * d3_inv[None] * voxel
    E = E.sum(axis=(1, 2, 3))
    return E


def coulomb_kspace(rho, cell_au, voxel):
    pass


def biot_savart(r0, r, j, cell=None, thresh=1.E-8):
    '''r...shape(N, ..., 3)'''
    # in atomic units using cgs convention for B field would be: µ0/4*pi = 1/c
    # here we use au w/o cgs : µ0/4*pi = 1/c**2
    # d = r0 - r
    d = mapping.distance_pbc(r, r0, cell=cell)  # r0 - r
    d3 = np.linalg.norm(d, axis=-1)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3 < thresh**3, 0.0, np.divide(1.0, d3))
    B = np.cross(j, d, axisa=-1, axisb=-1, axisc=-1) * d3_inv[:, None]
    return B / constants.c_au**2


def biot_savart_grid(r, j, pos_grid, voxel, cell=None, thresh=1.E-8):
    '''r...shape(3, ..., N)'''
    if cell is not None:
        raise NotImplementedError('coulomb_grid does not support periodic '
                                  'boundaries!')
    d = r[:, None, None, None]-pos_grid
    d3 = np.linalg.norm(d, axis=0)**3
    with np.errstate(divide='ignore'):
        d3_inv = np.where(d3 < thresh**3, 0.0, np.divide(1.0, d3))
    B = np.cross(j, d, axisa=0, axisb=0, axisc=0) * d3_inv[None] * voxel
    B = B.sum(axis=(1, 2, 3))
    return B / constants.c_au**2


def biot_savart_kspace(j, cell_vec_au, voxel):
    div, rot = divrot(j, cell_vec_au)

    # G != 0
    B1 = kspace.k_potential(rot[0], cell_vec_au)[1]
    B2 = kspace.k_potential(rot[1], cell_vec_au)[1]
    B3 = kspace.k_potential(rot[2], cell_vec_au)[1]
    B = np.array([B1, B2, B3]) / (4 * np.pi)
    # this 4 pi division should be done in kspace binary already?

    # G == 0 ???
    return B * voxel / constants.c_au**2
