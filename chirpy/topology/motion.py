# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import numpy as np
from ..classes.core import _PALARRAY
from ..physics.statistical_mechanics import time_correlation_function as tcf
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


# --- for _PALARRAY
def _func0(p,
           donor,
           acceptor,
           hydrogen,
           dist_crit,
           angle_crit,
           cell,
           ):

    return ishydrogenbond(
                    p,
                    donor,
                    acceptor,
                    hydrogen,
                    dist_crit=dist_crit,
                    angle_crit=angle_crit,
                    cell=cell,
                    )


def _cumulate_hydrogen_bonding_events(_H):
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
                         ])
    return segments


def _acf_c(h):
    segments = _cumulate_hydrogen_bonding_events(h)
    if len(segments) == 0:
        return np.zeros_like(h)
    B = np.mean([tcf(_s) for _s in segments],
                axis=0
                )
    return B / B[0]


def _acf_i(h):
    B = tcf(h)
    return B / B[0]


def hydrogen_bond_lifetime_analysis(positions, donor, acceptor, hydrogen,
                                    dist_crit=3.0,
                                    angle_crit=130,
                                    cell=None,
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
       mode:             intermittent/continuous

       returns:
       time correlation function (numpy array) averaged over all HB pairs
       found in the donor-acceptor-hydrogen pool (no_average=True: resolve
       individual pairs with shape (n_donors, n_acceptors)).
       '''

    # --- generate HB occurence trajectory (parallel run)
    # H = _PALARRAY(_func0, positions).run()
    H = _PALARRAY(_func0, positions,
                  donor=donor,
                  acceptor=acceptor,
                  hydrogen=hydrogen,
                  dist_crit=dist_crit,
                  angle_crit=angle_crit,
                  cell=cell
                  ).run()
    # print('Done with HB occurrence...')

    n_frames, n_donors, n_acceptors = H.shape
    _wH = np.array([_h
                    for _h in H.reshape((n_frames, -1)).T.astype(float)
                    if np.sum(_h) > 0])

    # --- correlate (parallel run)
    if mode == 'continuous':
        ACF = _PALARRAY(_acf_c, _wH).run()

    if mode == 'intermittent':
        ACF = _PALARRAY(_acf_i, _wH).run()

    if no_average:
        ACF_res = np.zeros((n_donors, n_acceptors, n_frames))
        # ACF_res[H.sum(axis=0) > min_length] = ACF
        ACF_res = ACF
        return ACF_res

    else:
        return np.average(ACF, axis=0, weights=ACF.sum(axis=-1))
