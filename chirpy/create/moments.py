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

import numpy as _np
import copy as _copy

from ..classes.core import _CORE
from ..topology import mapping as mp
from ..physics import classical_electrodynamics as ed
from ..physics import constants


class OriginGauge(_CORE):
    '''Object that processes moment trajectories from classes and
       converts between origin gauges.
       By moment data usually contains entries in
       the following order ("CPMD 4.1 convention"; see MOMENTS.style):
           origin, current dipole moment, magnetic dipole moment

       All data is in atomic units (position input in angstrom).
       '''
    def __init__(self, moments, cell=None):
        if (_name := moments.__class__.__name__) not in ['MOMENTS',
                                                         'MOMENTSFrame']:
            raise TypeError(f'unsupported object: {_name}')

        self.r_au = moments.data[:, :3] * constants.l_aa2au
        self.c_au = moments.data[:, 3:6]
        self.m_au = moments.data[:, 6:9]
        self.cell_au_deg = _copy.deepcopy(cell)
        if cell is not None:
            self.cell_au_deg[:3] *= constants.l_aa2au
        # --- heavy-atom gauge (not implemented)
        self.hag = False

    def __add__(self, other):
        if not _np.allclose(self.cell_au_deg, other.cell_au_deg):
            raise ValueError('the objects do not agree in cell')
        new = _copy.deepcopy(self)

        new.r_au = _np.concatenate((self.r_au, other.r_au))
        new.c_au = _np.concatenate((self.c_au, other.c_au))
        new.m_au = _np.concatenate((self.m_au, other.m_au))

        return new

    def switch_origin_gauge(self, origins_aa, assignment,
                            number_of_types=None):
        '''origins in angstrom'''
        _O = _copy.deepcopy(origins_aa) * constants.l_aa2au
        if (_n_o := len(origins_aa)) > len(self.r_au):
            raise ValueError('new number of origins cannot be greater than '
                             'the old one (assignment failure)')

        _R = _copy.deepcopy(self.r_au)
        _C = _copy.deepcopy(self.c_au)
        _M = _copy.deepcopy(self.m_au)
        _CELL = self.cell_au_deg
        lattice = mp.detect_lattice(_CELL)

        if lattice not in ['cubic', 'orthorhombic', 'tetragonal']:
            # --- convert positions to grid space
            _O = mp.get_cell_coordinates(_O, self.cell_au_deg)
            _R = mp.get_cell_coordinates(_R, self.cell_au_deg)
            _C = mp.get_cell_coordinates(_C, self.cell_au_deg)
            _M = mp.get_cell_coordinates(_M, self.cell_au_deg, angular=True)
            _CELL = _np.array([1., 1., 1., 90., 90., 90.])

        # --- process gauge-dependent properties
        # --- decompose according to assignment
        _Rd, _Cd, _Md = map(
                        lambda x: mp.dec(x, assignment, n_ind=number_of_types),
                        [_R, _C, _M]
                        )

        # --- --- magnetic dipole
        _Md_p = [ed.switch_magnetic_origin_gauge(*_tup, cell_au_deg=_CELL)
                 for _tup in zip(_Cd, _Md, _Rd, _O)]
        # --- --- electric dipole
        # --- --- multipole

        # --- sum parts
        _M = _np.array([_Md_p[_i].sum(axis=0) for _i in range(_n_o)])
        _C = _np.array([_Cd[_i].sum(axis=0) for _i in range(_n_o)])

        if lattice not in ['cubic', 'orthorhombic', 'tetragonal']:
            # --- back-convert position-dependent magnitudes to Cartesian space
            _O = mp.get_cartesian_coordinates(_O, self.cell_au_deg)
            _C = mp.get_cartesian_coordinates(_C, self.cell_au_deg)
            _M = mp.get_cartesian_coordinates(_M, self.cell_au_deg,
                                              angular=True)

        self.r_au = _O
        self.c_au = _C
        self.m_au = _M
