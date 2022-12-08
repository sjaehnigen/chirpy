# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2020-2022, The ChirPy Developers.
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
# ----------------------------------------------------------------------

__version__ = "0.24.3"


import sys

assert sys.version_info[:2] >= (3, 10), "Python version >= 3.10 required."

import multiprocessing as _mp
import multiprocessing.pool as _mpp

from . import config
from . import constants
from . import snippets
from . import classes
from . import create
from . import interface
from . import mathematics
from . import physics
from . import read
from . import topology
from . import visualise
from . import write
from . import external

# --- load important sub-modules
from .classes import *
from .create import *
from .snippets import *
from .snippets import _unpack_tuple
from .physics import spectroscopy


if __name__ == '__main__':
    _mp.set_start_method('spawn')


# import traceback
# def handle_exception(type, value, tb):
#     # length = 0
#     print(''.join(traceback.format_exception(type, value, tb)))
# sys.excepthook = handle_exception

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
