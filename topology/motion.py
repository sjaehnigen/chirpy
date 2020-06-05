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


def linear_momenta(velocities, wt, subset=slice(None), axis=-2):
    '''sum(velocities * wt)
       Use subset= to select atoms
       '''
    _wt = np.array(wt)[subset]
    _v = np.moveaxis(velocities, axis, 0)[subset]
    # _slc = (_sub,) + (len(_v.shape)-1) * (None,)

    linmoms = np.zeros_like(_v[0]).astype(float)
    for _iw, _w in enumerate(_wt):
        linmoms += _v[_iw] * _w

    return linmoms


def angular_momenta(positions, velocities, wt, subset=slice(None), axis=-2,
                    origin=np.zeros((3)), moI=False):
    '''sum(positions x velocities * wt)
       Use subset= to select atoms
       '''
    _wt = np.array(wt)[subset]
    _p = np.moveaxis(positions, axis, 0)[subset] - origin
    _v = np.moveaxis(velocities, axis, 0)[subset]
    # _slc = (_sub,) + (len(_p.shape)-1) * (None,)

    angmoms = np.zeros_like(_v[0]).astype(float)
    _moI = 0.0

    _r2 = np.linalg.norm(_p, axis=-1)**2
    for _iw, _ip, _iv, _ir2 in zip(_wt, _p, _v, _r2):
        angmoms += np.cross(_ip, _iv) * _iw
        _moI += _iw * _ir2

    if moI:
        return angmoms, _moI
    else:
        return angmoms
