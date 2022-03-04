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

import warnings
import numpy as np

from ..config import ChirPyWarning as _ChirPyWarning
from ..read.generators import _container
from ..read.coordinates import freeIterator as _freeIterator


def tinkermomentsReader(*args, gauge_origin_aa=[0., 0., 0.], columns='imddd',
                        **kwargs):
    '''Read tinker moment files (.*dip, *magdip, *ddip) and convert units.
       Distributed gauge origins specified via *.com file (optional).
       (arc format not supported)
       '''

    filetypes = {
            'pos': 'com',
            'cur': 'ddip',
            'mag': 'magdip',
            'dip': 'dip'
            }
    extensions = {
                    "ddip": "cur",
                    "dip": "dip",
                    "magdip": "mag",
                    "magdip_half": "mag",
                    "magdip_tot": "mag",
                    "com": "pos",
                    }

    if len(args) == 0:
        raise ValueError('tinkermomentsReader requires at least one file')
    try:
        _tmft = None  # to avoid flake warning only

        fn = {}
        for _fn in args:
            _tmft = _fn.split('.')[-1]
            if _tmft == "bz2":
                _tmft = _fn.split('.')[-2]
            if extensions[_tmft] in fn:
                raise ValueError('found more than one '
                                 f'*.{filetypes[extensions[_tmft]]} file')
            fn[extensions[_tmft]] = _fn

        units = kwargs.pop("units", 'default')
        if units == 'default':
            units = {
                'cur': 3*[('current_dipole', 'debye_ps')],
                'mag': 3*[('magnetic_dipole', 'debyeaa_ps')],
                'dip': 3*[('electric_dipole', 'debye')],
                'pos': 3*[('length', 'aa')],
                }
    except KeyError:
        raise ValueError('unknown tinker format: %s.' % _tmft)

    default = {}
    default['pos'] = np.array(gauge_origin_aa)
    # default['cur'] = np.array(3*[0.])
    # default['mag'] = np.array(3*[0.])
    # default['dip'] = np.array(3*[0.])

    def _get(_dict, key):
        value = _dict.get(key)
        if value is None:
            warnings.warn(f"missing tinker file *.{filetypes[key]}",
                          _ChirPyWarning, stacklevel=3)
        else:
            return value

    mom_a, reader_a, fn_a, args_a, kwargs_a = zip(*[
           (
               _im,
               _freeIterator,
               _fn,
               (),
               dict(units=units[_mom], columns=columns, **kwargs)
               )
           for _im, _mom in enumerate(['pos', 'cur', 'mag', 'dip'])  # order!
           if (_fn := _get(fn, _mom)) is not None
           ])
    mom_a = np.array(mom_a)

    # --- get id to pick only data columns from _freeIterator output
    _id = columns.index('d')

    data = None
    for _frame in _container(reader_a, fn_a, args_a=args_a, kwargs_a=kwargs_a):
        if data is None:
            try:
                data = np.tile(_frame[_id][0], (4, 1, 1)) * 0.0
            except IndexError:
                raise ValueError(f"expected columns \'{columns}\' in data")
            # --- fill in existing defaults
            data[0] = np.ones_like(data[0]) * default['pos']

        data[mom_a] = np.array(_frame[_id])

        try:
            yield np.transpose(data, axes=(1, 0, 2)).reshape(-1, 12).copy()

        except ValueError:
            raise ValueError('the given tinker files do not match')
