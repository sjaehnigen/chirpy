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


import os
import sys
import copy
import numpy as np
from chirpy.classes import quantum, volume

# --- Parametres ---
density = 'DENSITY-000001.cube'

current = 'CURRENT-000001-%d.cube'

state0 = 'C0-000001-S%02d.cube'
state1 = 'C1-000001-S%02d.cube'

current_state = 'CURRENT-000001-S%02d-%d.cube'


def _new_fn(_fn):
    return ''.join(_fn.split('.')[:-1]) + '-CROPPED.' + _fn.split('.')[-1]


def _np_fn(_fn):
    return ''.join(_fn.split('.')[:-1]) + '-CROPPED'


def _get_states(_dir=''):
    _n = 1
    while os.path.isfile(os.path.join(_dir, state0 % _n)):
        for _str in (
                state0 % _n,
                state1 % _n,
                current_state % (_n, 1),
                current_state % (_n, 2),
                current_state % (_n, 3)
                ):
            yield volume.ScalarField(os.path.join(_dir, _str))
        _n += 1


def _get_all(_dir=''):
    if os.path.isfile(os.path.join(_dir, density)):
        for _str in (
                density,
                current % 1,
                current % 2,
                current % 3
                ):
            yield volume.ScalarField(os.path.join(_dir, _str))
    _n = 1
    while os.path.isfile(os.path.join(_dir, state0 % _n)):
        for _str in (
                state0 % _n,
                state1 % _n,
                current_state % (_n, 1),
                current_state % (_n, 2),
                current_state % (_n, 3)
                ):
            yield volume.ScalarField(os.path.join(_dir, _str))
        _n += 1


def _get_fragments():
    _n = 0
    while os.path.isdir('fragment_%03d' % _n):
        for _st in _get_all(_dir='fragment_%03d' % _n):
            yield _st
        _n += 1


# --- Initial auto_crop to obtain crop value ---
print('Load Electronic System ...')
sys.stdout.flush()
_fn = density
_fn1 = current % 1
_fn2 = current % 2
_fn3 = current % 3
_sys = quantum.ElectronicSystem(_fn, _fn1, _fn2, _fn3)

_V = _sys.auto_crop(thresh=_sys.rho.threshold / 2)
print(' --> Crop %s' % _V)
sys.stdout.flush()

# --- Save ---
print('Save cropped Electronic System ...')
sys.stdout.flush()

out = copy.deepcopy(_sys.rho)
out.write(_new_fn(_fn))

out = copy.deepcopy(_sys.j)
out.write(_new_fn(_fn1), _new_fn(_fn2), _new_fn(_fn3))

# --- Load all CUBES and crop with value ---

print('Loop States ...')
for _sys in _get_states():
    print(_sys.fn)
    sys.stdout.flush()
    _sys.crop(_V)
    _sys.write(_new_fn(_sys.fn))
    np.save(_new_fn(_sys.fn)[:-5], _sys.data)

print('Loop Fragments ...')
for _sys in _get_fragments():
    print(_sys.fn)
    sys.stdout.flush()
    _sys.crop(_V)
    _sys.write(_new_fn(_sys.fn))
    np.save(_new_fn(_sys.fn)[:-5], _sys.data)


# --- AIM ---
# system.rho.aim()
# for i in range(system.rho.n_atoms):
#     fn_cube = 'atom-%d.cube'%i
#     system.rho.aim_atoms[i].write(fn_cube,comment2='AIM atom %d'%i)


print('Done.')
