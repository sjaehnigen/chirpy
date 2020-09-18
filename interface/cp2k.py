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
        step_n.append(float(buff[0]))
        time.append(float(buff[1]))
        kin.append(float(buff[2]))
        temp.append(float(buff[3]))
        pot.append(float(buff[4]))
        cqty.append(float(buff[5]))

    step_n = np.array(step_n)
    time = np.array(time)
    kin = np.array(kin)
    temp = np.array(temp)
    pot = np.array(pot)
    cqty = np.array(cqty)

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
