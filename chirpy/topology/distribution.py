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

from .mapping import distance_pbc, cell_volume
from ..mathematics.algebra import rotation_matrix
from ..classes.core import PALARRAY


def radial_distribution_function(positions,
                                 origins,
                                 cell=None,
                                 rng=(0.1, 10),
                                 bins=100,
                                 half_vector=None):
    '''Compute the normalised radial distribution function (RDF).
       Array of positions ((n_frames, n_particles, 3)) is evaluated
       against an array of origins ((n_frames, n_origins, 3)).
       Returns:
        r   - bin positions
        RDF - values
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
    _wg = positions.shape[0] * positions.shape[1] / volume

    global _func

    if half_vector is not None:
        def _func(x):
            return _rdf(
                     get_P(
                        positions,
                        origins[:, x],
                        cell=cell,
                        _hv=half_vector[:, x]
                        ),
                     rng,
                     bins
                    ) / _wg * 2

    else:
        def _func(x):
            return _rdf(
                     get_P(
                        positions,
                        origins[:, x],
                        cell=cell,
                        ),
                     rng,
                     bins
                    ) / _wg

    return np.linspace(*rng, bins),\
        np.mean(PALARRAY(_func, range(n_O)).run(), axis=0)


def rdf(*args, **kwargs):
    return radial_distribution_function(*args, **kwargs)
