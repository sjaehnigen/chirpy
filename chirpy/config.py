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

import sys
import importlib
import warnings
from IPython import get_ipython
import multiprocessing
import platform

from .version import version

__pal_n_cores__ = max(multiprocessing.cpu_count()//2, 1)
__verbose__ = True
__os__ = platform.system()


class ChirPyWarning(UserWarning):
    pass


def version_info():
    return tuple([int(_v) for _v in version.split('.')])


def _reload_modules():
    # --- apply changes to loaded modules
    modules = tuple(sys.modules.values())
    for module in modules:
        if 'chirpy.' in (module_name := module.__name__) \
         and 'config' not in module_name:
            importlib.reload(module)


def set_pal_n_cores(s, reload_modules=True):
    global __pal_n_cores__
    __pal_n_cores__ = int(s)
    if __verbose__:
        warnings.warn(f'__pal_n_cores__ set to {__pal_n_cores__}',
                      ChirPyWarning,
                      stacklevel=2)
    if reload_modules:
        _reload_modules()


def set_verbose(s, reload_modules=True):
    '''Enable/disable chirpy runtime verbosity.'''
    global __verbose__
    # if __verbose__ and not s:
    #     warnings.warn(f'__verbose__ set to {bool(s)}',
    #                   ChirPyWarning,
    #                   stacklevel=2)
    __verbose__ = bool(s)
    if reload_modules:
        _reload_modules()


# --- check if run in ipython/jupyter notebook
if get_ipython() is not None:
    if "IPKernelApp" in get_ipython().config:
        # warnings.warn('jupyter: enabling chirpy verbosity', stacklevel=2)
        if not __verbose__:
            set_verbose(True)
