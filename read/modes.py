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


import numpy as np
import warnings as _warnings

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

    # --- mw should be convention be False!
    mw = kwargs.get('mw', False)
    if mw:
        _warnings.warn('Assuming mass-weighted coordinates in XVIBS.',
                       stacklevel=2)
        masses_amu = constants.numbers_to_masses(numbers)
        modes /= np.sqrt(masses_amu)[None, :, None]

    return n_atoms, numbers, pos_aa, n_modes, freqs_cgs, modes
