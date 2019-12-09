#!/usr/bin/env python3
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
import warnings as _warnings

from ..interface import orca
from ..physics import constants


def xvibsReader(fn, **kwargs):
    with open(fn, 'r') as f:
        inbuffer = f.read()
    pos_natoms = inbuffer.index('NATOMS')
    pos_coords = inbuffer.index('COORDINATES')
    pos_freqs = inbuffer.index('FREQUENCIES')
    pos_modes = inbuffer.index('MODES')
    pos_end = inbuffer.index('&END')

    n_atoms = int(inbuffer[pos_natoms+7:pos_coords].strip())
    numbers = list()
    pos_aa = list()

    for line in inbuffer[pos_coords+12:pos_freqs].strip().split('\n'):
        tmp = line.split()
        numbers.append(int(tmp[0]))
        pos_aa.append([float(e) for e in tmp[1:4]])
    pos_aa = np.array(pos_aa)
    sec_freqs = inbuffer[pos_freqs+12:pos_modes].strip().split('\n')
    n_modes = int(sec_freqs[0])
    freqs_cgs = list()
    for line in sec_freqs[1:]:
        freqs_cgs.append(float(line))
    sec_modes = inbuffer[pos_modes+5:pos_end].strip().split('\n')
    modes = np.zeros((n_modes*n_atoms, 3))
    for i, line in enumerate(sec_modes):
        tmp = line.split()
        modes[i] = np.array([float(e) for e in tmp])
    modes = modes.reshape((n_modes, n_atoms, 3))

    # --- DEBUG corner
    # manually switch mw ON/OFF (it should be clear from the file format)
    mw = kwargs.get('mw', False)
    if mw:
        _warnings.warn('Assuming mass-weighted coordinates in XVIBS.',
                       UserWarning)
        masses_amu = constants.numbers_to_masses(numbers)
        modes /= np.sqrt(masses_amu)[None, :, None] *\
            np.sqrt(constants.m_amu_au)
    # ---

    return n_atoms, numbers, pos_aa, n_modes, freqs_cgs, modes


def orcaReader(fn):
    '''Reads ORCA 4.3 files. Currently supported:
        *.hess
        '''
    fmt = fn.split('.')[-1]
    if fmt == "hess":
        return orca.read_hessian_file(fn)
    else:
        raise ValueError('Unknown file format.')
