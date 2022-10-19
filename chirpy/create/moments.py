# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2020-2022, The ChirPy Developers.
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

import numpy as _np
import copy as _copy

from ..classes.core import CORE as _CORE
from ..topology import mapping as mp
from ..physics import classical_electrodynamics as ed
from .. import constants


class OriginGauge(_CORE):
    '''Object that processes moment trajectories from classes and
       converts between origin gauges.

       Data are numpy arrays of shape ([n_frames], n_units, dim).

       All data is in atomic units (position input in angstrom).
       '''
    def __init__(self,
                 origin_aa=None,
                 current_dipole_au=None,
                 magnetic_dipole_au=None,
                 electric_dipole_au=None,
                 charge_au=None,
                 cell_aa_deg=None,
                 ):

        # --- parse moments
        self.c_au = _copy.deepcopy(current_dipole_au)
        self.m_au = _copy.deepcopy(magnetic_dipole_au)
        self.d_au = _copy.deepcopy(electric_dipole_au)

        if len(self.c_au.shape) == 3:  # --- trajectory
            self._workax = 1
            self.n_frames, self.n_units, three = self.c_au.shape
        elif len(self.c_au.shape) == 2:  # --- frame
            self._workax = 0
            self.n_units, three = self.c_au.shape

        self._set = ''.join([_i for _i in 'cmd'
                             if getattr(self, f'{_i}_au') is not None])
        if self._set == '':
            raise ValueError('could not find moment data. Did nothing.')
        if 'm' not in self._set and 'c' not in self._set:
            raise NotImplementedError('OriginGauge requires at least current '
                                      'and magnetic dipole moments.')

        # --- parse auxiliary data
        self.r_au = _copy.deepcopy(origin_aa * constants.l_aa2au)
        if isinstance(charge_au, (int, float)):
            self.q_au = _np.ones_like(self.r_au[..., 0]) * charge_au
        else:
            self.q_au = _np.ones_like(self.r_au[..., 0]) * _np.array(charge_au)
            # self.q_au = _copy.deepcopy(charge_au)

        # --- check for required gauge lever and weights
        if any(_i in self._set for _i in 'cm') and self.r_au is None:
            raise ValueError(f'moment set \'{self._set }\' requires argument:'
                             'origin_aa')
        if 'm' in self._set and self.c_au is None:
            raise ValueError(f'moment set \'{self._set }\' requires argument:'
                             'current_dipole_au')
        if self.q_au is None:
            raise ValueError(f'moment set \'{self._set }\' requires argument:'
                             'charge_au')

        # --- periodic boundaries
        self.cell_au_deg = _copy.deepcopy(cell_aa_deg)
        if cell_aa_deg is not None:
            self.cell_au_deg[:3] *= constants.l_aa2au

    def __add__(self, other):
        # if not _np.allclose(self.cell_au_deg, other.cell_au_deg):
        #     raise ValueError('the objects do not agree in cell')
        new = _copy.deepcopy(self)
        new.r_au = _np.concatenate((self.r_au, other.r_au), axis=self._workax)
        new.c_au = _np.concatenate((self.c_au, other.c_au), axis=self._workax)
        new.m_au = _np.concatenate((self.m_au, other.m_au), axis=self._workax)
        new.q_au = _np.concatenate((self.q_au, other.q_au), axis=self._workax)
        if 'd' in self._set:
            new.d_au = _np.concatenate((self.d_au, other.d_au),
                                       axis=self._workax)

        return new

    @staticmethod
    def time_derivative(data, dt=1.):
        '''data array of shape (n_frames, [n_units, 3])'''
        return _np.gradient(data, axis=0) / dt

    def shift_origin_gauge(self, origins_aa, assignment=None,
                           number_of_types=None):
        '''origins in angstrom of shape ([n_frames], n_origins, 3)'''
        _O = _copy.deepcopy(origins_aa) * constants.l_aa2au
        # --- add np.newaxis for n_units
        _O = _O[None].swapaxes(0, -2)

        if (n_O := origins_aa.shape[-2]) > self.n_units:
            raise ValueError('number of origins must be less or equal the '
                             f'number of units: {n_O} > {self.n_units}')
        if assignment is None:
            if n_O == self.n_units:
                assignment = _np.arange(self.n_units)
            else:
                raise ValueError('changed number of origins requires '
                                 'assignment argument')

        _R = _copy.deepcopy(self.r_au)
        _C = _copy.deepcopy(self.c_au)
        _M = _copy.deepcopy(self.m_au)
        _D = _copy.deepcopy(self.d_au)
        _Q = _copy.deepcopy(self.q_au)

        if (lattice := mp.detect_lattice(self.cell_au_deg)) is not None:
            _CELL = self.cell_au_deg
        else:
            _CELL = None

        if lattice not in ['cubic', 'orthorhombic', 'tetragonal', None]:
            # --- convert positions to grid space
            _O = mp.get_cell_coordinates(_O, self.cell_au_deg)
            _R = mp.get_cell_coordinates(_R, self.cell_au_deg)
            _C = mp.get_cell_coordinates(_C, self.cell_au_deg)
            _M = mp.get_cell_coordinates(_M, self.cell_au_deg, angular=True)
            _CELL = _np.array([1., 1., 1., 90., 90., 90.])
            if 'd' in self._set:
                _D = mp.get_cell_coordinates(_D, self.cell_au_deg)

        # --- process gauge-dependent properties
        # --- decompose according to assignment
        _Rd, _Cd, _Md, _Qd = map(
               lambda x: mp.dec(x, assignment, n_ind=number_of_types,
                                axis=self._workax),
               [_R, _C, _M, _Q]
               )
        # --- --- magnetic dipole
        _Md_p = [ed.shift_magnetic_origin_gauge(*_tup, cell_au_deg=_CELL)
                 for _tup in zip(_Cd, _Md, _Rd, _O)]
        # --- --- electric dipole
        if 'd' in self._set:
            _Dd = mp.dec(_D, assignment, n_ind=number_of_types,
                         axis=self._workax)
            _Dd_p = [ed.shift_electric_origin_gauge(*_tup, cell_au_deg=_CELL)
                     for _tup in zip(_Qd, _Dd, _Rd, _O)]
            _D = _np.array([_Dd_p[_i].sum(axis=self._workax)
                            for _i in range(n_O)])

        # --- --- multipole

        # --- sum parts
        _M = _np.array([_Md_p[_i].sum(axis=self._workax) for _i in range(n_O)])
        _C = _np.array([_Cd[_i].sum(axis=self._workax) for _i in range(n_O)])
        _Q = _np.array([_Qd[_i].sum(axis=self._workax) for _i in range(n_O)])
        # --- undo expand_dims()
        _O = _np.take(_O, 0, axis=-2)

        if lattice not in ['cubic', 'orthorhombic', 'tetragonal', None]:
            # --- back-convert position-dependent magnitudes to Cartesian space
            _O = mp.get_cartesian_coordinates(_O, self.cell_au_deg)
            _C = mp.get_cartesian_coordinates(_C, self.cell_au_deg)
            _M = mp.get_cartesian_coordinates(_M, self.cell_au_deg,
                                              angular=True)
            if 'd' in self._set:
                _D = mp.get_cartesian_coordinates(_D, self.cell_au_deg)

        if self._workax != 0:
            _O = _O.swapaxes(0, self._workax)
            _C = _C.swapaxes(0, self._workax)
            _M = _M.swapaxes(0, self._workax)
            _Q = _Q.swapaxes(0, self._workax)
            if 'd' in self._set:
                _D = _D.swapaxes(0, self._workax)

        self.r_au = _O
        self.c_au = _C
        self.m_au = _M
        self.d_au = _D
        self.q_au = _Q
        self.n_units = n_O
