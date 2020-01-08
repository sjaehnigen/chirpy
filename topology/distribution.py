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


import numpy as np

from .mapping import detect_lattice
from ..mathematics.algebra import rotation_matrix


def radial_distribution_function(DS, DO, cell_au_deg=None, **kwargs):
    '''DS/O ... source/origin data
       '''
    n_frames, n_O, three = DO.shape
    h_v = kwargs.get('half_vector', None)  # BETA: still testing this option
    del kwargs['half_vector']

    def _rdf(_P, rng=(0.1, 10), bins=100):
        '''RDF kernel.
           pos of shape (n_frames,n_particles,3)
           ref integer of reference particle'''

        R = np.linspace(rng[0], rng[1], bins)
        rdf = np.histogram(
                _P,
                bins=bins,
                density=False,
                range=rng
                )[0].astype(float)

        # --- divide by shell volume
        rdf /= 4. * np.pi * R**2 * (R[1] - R[0])

        return rdf

    def get_P(s, o, _hv=None):
        _P = s - o[:, None]
        if detect_lattice(cell_au_deg) is not None:
            _P -= np.around(_P/cell_au_deg) * cell_au_deg

        if _hv is not None:  # beta
            ind = np.array([
                    np.tensordot(
                        rotation_matrix(
                            _v,
                            np.array([0.0, 0.0, 1.0])
                        ),
                        _p,
                        axes=([1], [1])
                    ).swapaxes(0, 1) for _v, _p in zip(_hv, _P)
            ])
            _P = _P[ind[:, :, 2] > 0]  # returns flattened first 2 dims
        return np.linalg.norm(_P, axis=-1)  # .flatten() #auto-flattening?

    # --- norm to n_frames and density
    _wg = n_O * DS.shape[0] * DS.shape[1] / np.prod(cell_au_deg)

    if h_v is not None:
        return np.sum([_rdf(get_P(DS, DO[:, _o], _hv=h_v[:, _o]), **kwargs)
                       for _o in range(n_O)], axis=0) / _wg * 2

    else:
        return np.sum([_rdf(get_P(DS, DO[:, _o]), **kwargs)
                       for _o in range(n_O)], axis=0) / _wg


def rdf(*args, **kwargs):
    return radial_distribution_function(*args, **kwargs)
