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

import sys
import multiprocessing as _mp
import multiprocessing.pool as _mpp

from . import config
from . import constants
from . import snippets
from . import version
from . import classes
from . import create
from . import interface
from . import mathematics
from . import physics
from . import read
from . import topology
from . import visualise
from . import write

# --- load important sub-modules
from .classes import *
from .create import *

if __name__ == '__main__':
    _mp.set_start_method('spawn')


assert sys.version_info[:2] >= (3, 8), "Python version >= 3.8 required."

# --- update multiprocessing
#    ( https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm)

def istarmap(self, func, iterable, chunksize=1):
    '''starmap-version of imap
    '''
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = _mpp.Pool._get_tasks(func, iterable, chunksize)
    result = _mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          _mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


_mpp.Pool.istarmap = istarmap


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values

        Taken from:
        https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/arraysetops.py
        Copyright (c) 2005-2021, NumPy Developers.

        """
    if len(x) == 1:
        return x[0]
    else:
        return x
