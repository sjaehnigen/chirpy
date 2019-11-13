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


def read_hessian_file(fn):
    '''This is an antiquated reader for Orca .hess files'''

    with open(fn, 'r') as f:
        inbuffer = f.read()
    if inbuffer.strip().split('\n')[0] != '$orca_hessian_file':
        print(inbuffer.strip().split('\n')[1])
        raise Exception('No ORCA format?!')

    # pos_hessian = inbuffer.index('$hessian')
    pos_freqs = inbuffer.index('$vibrational_frequencies')
    pos_modes = inbuffer.index('$normal_modes')
    pos_coords = inbuffer.index('$atoms')
    pos_end = inbuffer.index('$end')

    freqs = list()
    sec_freqs = inbuffer[pos_freqs+25: pos_modes].strip().split('\n')
    n_freqs = int(sec_freqs[0])
    print('Found %s vibrational frequencies.' % n_freqs)
    freqs = np.array([float(_l.split()[1]) for _l in sec_freqs[1: n_freqs+1]])

    pos_au = list()
    symbols = list()
    sec_coords = inbuffer[pos_coords+7: pos_end].strip().split('\n')
    n_atoms = int(sec_coords[0])
    print('Found %s atoms.' % n_atoms)

    symbols = [_l.split()[0] for _l in sec_coords[1: n_atoms+1]]
    pos_au = np.array([list(map(float, _l.split()[2:]))
                       for _l in sec_coords[1: n_atoms+1]])

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
                print('Read %s normal modes.' % n_freqs)
                break
    modes_res = modes.reshape((n_freqs, n_atoms, 3))

    return symbols, pos_au, freqs, modes, modes_res
