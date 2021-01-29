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
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
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

from chirpy.classes import trajectory
from chirpy.physics import constants
from chirpy.physics.statistical_mechanics import spectral_density

from chirpy.classes import _PALARRAY

# this is essentially ft_colvar_cor (but without the sum)
# to avoid memory issues, localisation etc has to happen here as well?
# (don't keep ft_colvar_cor)


def _func(x0, x1):
    _omega, _S, _R = spectral_density(x0, x1, ts=1, flt_pow=-1, mode='A')
    return _S.sum(axis=-1)


def vv_variance(vel):
    _n_frames, _n_atoms, _dim = vel.shape
    vel = vel.reshape((_n_frames, _n_atoms * _dim))

    JOB = _PALARRAY(_func, vel.T, repeat=2, upper_triangle=True, n_cores=6)
    S = JOB.run()

    # --- see comment in spectral_density module for explanation of prefactors
    return S * 2 * np.pi / _n_frames


_dir = '/home/jaehnigen/trajectories_nobackup/l-alanine-crystal/P212121_2x1x2'\
       '/production/'
_load = trajectory.XYZ(_dir + 'traj.xyz', range=(0, 64, 100000))

TRAJ = _load.expand()

print(TRAJ.data.shape)
_n_frames, _n_atoms, _dim = TRAJ.data.shape

# ts = 4
POW = vv_variance(TRAJ.vel_au, n_cores=4)

masses_au = constants.symbols_to_masses(sorted(3 * TRAJ.symbols)) \
        * constants.m_amu_au

effective_mass = constants.k_B_au * 337 / np.diag(POW) / masses_au

np.set_printoptions(precision=2, linewidth=100)

print('EFFECTIVE MASS')
print('1.00000 = %5.5f ?' % np.mean(effective_mass))
# print(effective_mass)

# NEXT: localisation
