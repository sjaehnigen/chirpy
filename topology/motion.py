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


def linear_momenta(velocities, wt, **kwargs):
    '''sum(velocities * wt)
       Use subset= to select atoms
       '''
    _sub = kwargs.get('subset', slice(None))
    _axis = kwargs.get("axis", -2)
    _wt = np.array(wt)[_sub]
    _v = np.moveaxis(velocities, _axis, 0)[_sub]
    # _slc = (_sub,) + (len(_v.shape)-1) * (None,)

    linmoms = np.zeros_like(_v[0]).astype(float)
    for _iw, _w in enumerate(_wt):
        linmoms += _v[_iw] * _w

    return linmoms


def angular_momenta(positions, velocities, wt, **kwargs):
    '''sum(positions x velocities * wt)
       Use subset= to select atoms
       '''
    _sub = kwargs.get('subset', slice(None))
    _axis = kwargs.get("axis", -2)
    _o = np.array(kwargs.get("origin", 3 * [0]))
    _wt = np.array(wt)[_sub]
    _p = np.moveaxis(positions, _axis, 0)[_sub] - _o
    _v = np.moveaxis(velocities, _axis, 0)[_sub]
    # _slc = (_sub,) + (len(_p.shape)-1) * (None,)

    angmoms = np.zeros_like(_v[0]).astype(float)
    for _iw, _w in enumerate(_wt):
        angmoms += np.cross(_p[_iw], _v[_iw]) * _w

    return angmoms
