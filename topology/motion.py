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
from ..classes import core
from ..physics.statistical_mechanics import time_correlation_function
from .mapping import ishydrogenbond


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


def hydrogen_bond_lifetime_analysis(positions, donor, acceptor, hydrogen,
                                    dist_crit=3.0,
                                    angle_crit=130,
                                    cell=None,
                                    min_length=1,
                                    mode='intermittent',
                                    no_average=False):
    '''Compute auto-correlation function of hydrogen bond occurrence between
       donor and acceptor (heavy atoms).

       positions:        position array of shape (n_frames, n_atoms, 3)
       donor/acceptor:   atom indices of heavy atoms donating/accepting HBs
       hydrogen:         indices of the (sub)set of hydrogen atoms

       dist_crit:        float in units of positions
       angle_crit:       float in degrees
       cell:             a b c al be ga
       min_length:       minimum period in frames to count HB connection
                         (continuous mode only)
       mode:             intermittent/continuous

       returns:
       time correlation function (numpy array) averaged over all HB pairs
       found in the donor-acceptor-hydrogen pool (no_average=True: resolve
       individual pairs with shape (n_donors, n_acceptors)).
       '''

    def cumulate_hydrogen_bonding_events(_H, min_length):
        '''Split timeline into individual HB events and move them to t=0
           (zero padding).

           min_length:       minimum period in frames to count HB connection
           '''
        n_frames = _H.shape[0]
        _diff = np.diff(_H, axis=0, prepend=0)
        _edges = np.argwhere(_diff == 1).flatten()

        segments = np.array([np.pad(_s, (0, n_frames - len(_s)))
                             for _s in np.split(_H,
                                                axis=0,
                                                indices_or_sections=_edges)[1:]
                             if np.sum(_s) >= min_length])
        return segments

    global func0
    global _acf_c
    global _acf_i

    def func0(p):
        return ishydrogenbond(
                        p,
                        donor,
                        acceptor,
                        hydrogen,
                        dist_crit=dist_crit,
                        angle_crit=angle_crit,
                        cell=cell
                        )

    def _acf_c(h):
        segments = cumulate_hydrogen_bonding_events(h, min_length)
        if len(segments) == 0:
            return np.zeros_like(h)
        B = np.mean([time_correlation_function(_s) for _s in segments],
                    axis=0
                    )
        return B / B[0]

    def _acf_i(h):
        B = time_correlation_function(h)
        return B / B[0]

    # --- generate HB occurence trajectory (parallel run)
    H = core._PALARRAY(func0, positions).run()
    # print('Done with HB occurrence...')

    n_frames, n_donors, n_acceptors = H.shape
    _wH = np.array([_h
                    for _h in H.reshape((n_frames, -1)).T.astype(float)
                    if np.sum(_h) > min_length])

    # --- correlate (parallel run)
    if mode == 'continuous':
        ACF = core._PALARRAY(_acf_c, _wH).run()

    if mode == 'intermittent':
        ACF = core._PALARRAY(_acf_i, _wH).run()
    # print('Done with ACF...')

    if no_average:
        ACF_res = np.zeros((n_donors, n_acceptors, n_frames))
        ACF_res[H.sum(axis=0) > min_length] = ACF
        return ACF_res

    else:
        return np.average(ACF, axis=0, weights=ACF.sum(axis=-1))
