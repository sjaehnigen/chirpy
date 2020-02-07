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
