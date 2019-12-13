#!/usr/bin/env python
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


def xvibsWriter(filename, n_atoms, numbers, pos_aa, freqs, modes):
    obuffer = '&XVIB\n NATOMS\n %d\n COORDINATES\n' % n_atoms
    for n, r in zip(numbers, pos_aa):
        obuffer += ' %d  %16.12f  %16.12f  %16.12f\n' % tuple([n] + list(r))
    obuffer += ' FREQUENCIES\n %d\n' % len(freqs)
    for f in freqs:
        obuffer += ' %16.12f\n' % f
    obuffer += ' MODES\n'
    n_modes, atoms, three = modes.shape
    modes = modes.reshape((n_modes * atoms, 3))
    for mode in modes:
        obuffer += ' %16.12f  %16.12f  %16.12f\n' % tuple(mode)
    obuffer += '&END\n'

    with open(filename, 'w') as f:
        f.write(obuffer)
