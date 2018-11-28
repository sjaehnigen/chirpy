#!/usr/bin/env python3
import numpy as np
from physics import constants

def xvibsWriter(filename, n_atoms, numbers, coordinates, frequencies, modes):
    '''Module by Arne Scherrer'''
    obuffer = '&XVIB\n NATOMS\n %d\n COORDINATES\n'%n_atoms
    for n, r in zip(numbers, coordinates):
        obuffer += ' %d  %16.12f  %16.12f  %16.12f\n'%tuple([n]+list(r*constants.l_aa2au))
    obuffer += ' FREQUENCIES\n %d\n'%len(frequencies)
    for f in frequencies:
        obuffer += ' %16.12f\n'%f
    obuffer += ' MODES\n'
    n_modes, atoms, three = modes.shape
    modes = modes.reshape((n_modes*atoms, 3))
    for mode in modes:
        obuffer += ' %16.12f  %16.12f  %16.12f\n'%tuple(mode)
    obuffer += '&END\n'
    with open(filename, 'w') as f:
        f.write(obuffer)
