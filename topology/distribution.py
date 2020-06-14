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

from .mapping import distance_pbc, cell_volume
from ..mathematics.algebra import rotation_matrix


def radial_distribution_function(positions,
                                 origins,
                                 cell=None,
                                 rng=(0.1, 10),
                                 bins=100,
                                 half_vector=None):
    '''Compute the normalised radial distribution function (RDF).
       Array of positions ((n_frames, n_particles, 3)) is evaluated
       against an array of origins ((n_frames, n_origins, 3)).
       '''
    n_frames, n_O, three = origins.shape

    if cell is not None:
        volume = cell_volume(cell)
    else:
        volume = 1.0

    def _rdf(_P, rng, bins):
        '''RDF kernel.
           _P … positions of shape (n_frames, n_particles, 3)
           '''

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

    def get_P(s, o, _hv=None, cell=cell):
        _P = distance_pbc(o[:, None], s, cell=cell)

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
    _wg = n_O * positions.shape[0] * positions.shape[1] / volume

    if half_vector is not None:
        return np.sum([_rdf(get_P(positions,
                                  origins[:, _o],
                                  cell=cell,
                                  _hv=half_vector[:, _o]),
                            rng,
                            bins)
                       for _o in range(n_O)], axis=0) / _wg * 2

    else:
        return np.sum([_rdf(get_P(positions,
                                  origins[:, _o],
                                  cell=cell),
                            rng,
                            bins)
                       for _o in range(n_O)], axis=0) / _wg


def rdf(*args, **kwargs):
    return radial_distribution_function(*args, **kwargs)
