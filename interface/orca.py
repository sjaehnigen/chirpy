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
import warnings


def orcaReader(fn):
    '''Reads ORCA 4.3 files. Currently supported:
        *.hess
        '''
    fmt = fn.split('.')[-1]
    if fmt == "hess":
        return read_hessian_file(fn)
    else:
        raise NotImplementedError('Unknown file format!')


def read_hessian_file(fn):
    '''This is an antiquated reader for Orca .hess files'''

    with open(fn, 'r') as f:
        inbuffer = f.read()
    if inbuffer.strip().split('\n')[0] != '$orca_hessian_file':
        print(inbuffer.strip().split('\n')[1])
        raise ValueError('Cannot read file. No ORCA format?')

    # pos_hessian = inbuffer.index('$hessian')
    pos_freqs = inbuffer.index('$vibrational_frequencies')
    pos_modes = inbuffer.index('$normal_modes')
    pos_coords = inbuffer.index('$atoms')
    pos_deriv = inbuffer.index('$dipole_derivatives')
    pos_spec = inbuffer.index('ir_spectrum')
    pos_end = inbuffer.index('$end')

    sec_freqs = inbuffer[pos_freqs+25: pos_modes].strip().split('\n')
    n_freqs = int(sec_freqs[0])
    warnings.warn('ORCA interface: Found %d vibrational frequencies.'
                  % n_freqs,
                  stacklevel=2)
    omega_cgs = np.array([float(_l.split()[1])
                         for _l in sec_freqs[1: n_freqs+1]])

    pos_au = list()
    symbols = list()
    # --- read $atoms
    sec_coords = inbuffer[pos_coords+7: pos_end].strip().split('\n')
    n_atoms = int(sec_coords[0])

    symbols = [_l.split()[0] for _l in sec_coords[1: n_atoms+1]]
    pos_au = np.array([list(map(float, _l.split()[2:]))
                       for _l in sec_coords[1: n_atoms+1]])

    # --- read $dipole_derivatives (ToDo: verify unit)
    sec_deriv = inbuffer[pos_deriv+20: pos_end].strip().split('\n')
    deriv_au = np.array([list(map(float, _l.split()))
                         for _l in sec_deriv[1: n_freqs+1]])

    # --- read $ir_spectrum (ToDo: verify unit)
    sec_spec = inbuffer[pos_spec+13: pos_end].strip().split('\n')
    spec = np.array([list(map(float, _l.split()))
                     for _l in sec_spec[1: n_freqs+1]])

    tmp = list()
    modes = np.zeros((n_freqs, n_freqs))
    sec_modes = inbuffer[pos_modes+14: pos_coords].strip().split('\n')
    col_modes = len(sec_modes[1].split())
    blk_modes = int(np.ceil(n_freqs/col_modes))
    for block in range(blk_modes):
        for col in range(col_modes):
            i = int(block*col_modes+col)
            tmp[:] = []
            for line in range(2, n_freqs+2):
                tmp.append(sec_modes[line+block*(n_freqs+1)].split()[col+1])
            modes[i] = np.array([float(e) for e in tmp])
            if i >= n_freqs-1:
                break
    modes_res = modes.reshape((n_freqs, n_atoms, 3))

    data = {}
    data['symbols'] = symbols
    data['pos_au'] = pos_au
    data['omega_cgs'] = omega_cgs
    data['modes'] = modes_res
    data['T'] = spec[:, 2:]  # in sqrt(km/mol)
    data['T**2'] = spec[:, 1]  # in km/mol
    data['APT_au'] = deriv_au.reshape((n_atoms, 3, 3))  # dµ / dQ_a

    return data
