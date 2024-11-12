# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2024, The ChirPy Developers.
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

import warnings
import numpy as np


def parse_restart_file(fn):
    def _collect(_iter):
        COL = {}
        COL['KEYWORDS'] = []
        for _l in _iter:
            if '&END' in _l:
                break
            if "&" in _l:
                COL[_l[1:].upper()] = _collect(_iter)
            else:
                COL['KEYWORDS'].append(_l)
        return COL

    with open(fn, 'r') as _f:
        _iter = (_l.strip() for _l in _f)
        CONTENT = _collect(_iter)

    if len(CONTENT) == 0:
        raise ValueError(f'Could not read file {fn}! Is this CP2K?')

    if 'GLOBAL' not in CONTENT or 'FORCE_EVAL' not in CONTENT:
        warnings.warn('Invalid or incomplete CP2K input/restart file!',
                      RuntimeWarning, stacklevel=2)
    try:
        CONTENT['FORCE_EVAL']['SUBSYS']['COORD']
    except KeyError:
        warnings.warn('Could not find atom coordinates in file!',
                      stacklevel=2)

    return CONTENT


def read_ener_file(fn):
    with open(fn, 'r') as f:
        f.readline()[1:]   # title
        steps = f.readlines()

    kin = list()
    pot = list()
    temp = list()
    cqty = list()
    time = list()
    step_n = list()

    for step in steps:
        buff = step.split()
        if buff[0][0] == '#':
            continue
        step_n.append(float(buff[0]))
        time.append(float(buff[1]))
        kin.append(float(buff[2]))
        temp.append(float(buff[3]))
        pot.append(float(buff[4]))
        cqty.append(float(buff[5]))

    step_n = np.array(step_n)
    step_n = np.arange(len(step_n))  # overwrite
    time = np.array(time).astype(float)
    kin = np.array(kin).astype(float)
    temp = np.array(temp).astype(float)
    pot = np.array(pot).astype(float)
    cqty = np.array(cqty).astype(float)

#    kin_avg = np.average(kin)
#    pot_avg = np.average(pot)
#    cqty_avg= np.average(cqty)

    return step_n, time, temp, kin, pot, cqty


def read_tot_dipole_file(fn):
    """returns total dipole moments in a.u."""

    dat = np.genfromtxt(fn,
                        dtype=None,
                        comments='#',
                        usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                        deletechars='_',
                        autostrip=True
                        )
    dip_au = dat[:, :3]
    dip_de = dat[:, 3:6]
    ddip_au = dat[:, 6:]

    return dip_au, dip_de, ddip_au
