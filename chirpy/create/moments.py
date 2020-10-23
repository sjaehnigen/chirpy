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

from ..classes import _CORE
from ..topology import mapping
from ..physics import classical_electrodynamics as ed


class OriginGauge(_CORE):
    '''Object that processes moment trajectories from classes and
       converts between origin gauges.
       By moment data usually contains entries in
       the following order ("CPMD 4.1 convention"; see MOMENTS.style):
           origin, current dipole moment, magnetic dipole moment

       It is recommended that all data is in atomic units.
       '''
    def __init__(self, moments, cell=None):
        if (_name := moments.__class__.__name__) not in ['MOMENTS',
                                                         'MOMENTSFrame']:
            raise TypeError(f'unsupported object: {_name}')

        self.r = moments.data[:, :3]
        self.c = moments.data[:, 3:6]
        self.m = moments.data[:, 6:9]
        self.cell = cell
        # --- heavy-atom gauge (not implemented)
        self.hag = False

    def __add__(self, other):
        if not _np.allclose(self.cell, other.cell):
            raise ValueError('the objects do not agree in cell')
        new = _copy.deepcopy(self)

        new.r = _np.concatenate((self.r, other.r))
        new.c = _np.concatenate((self.c, other.c))
        new.m = _np.concatenate((self.m, other.m))

        return new

    def switch_origin_gauge(self, origins, assignment, number_of_types=None):
        if (_n_origins := len(origins)) > len(self.r):
            raise ValueError('new number of origins cannot be greater than '
                             'the old one (assignment failure)')
        # --- decompose according to assignment
        _rd, _cd, _md = map(
                lambda x: mapping.dec(x, assignment, n_ind=number_of_types),
                [self.r, self.c, self.m]
                )

        # --- process gauge-dependent properties
        # --- --- magnetic dipole
        _md_p = [ed.switch_magnetic_origin_gauge(_c, _m, _r, _o,
                                                 cell_au_deg=self.cell)
                 for _o, _r, _c, _m in zip(origins, _rd, _cd, _md)]
        # --- --- electric dipole
        # --- --- multipole

        # --- sum contributions
        self.r = origins
        self.c = _np.array([_cd[_i].sum(axis=0) for _i in range(_n_origins)])
        self.m = _np.array([_md_p[_i].sum(axis=0) for _i in range(_n_origins)])
