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


from itertools import islice
import numpy as np
import bz2 as _bz2
from tqdm import tqdm

from .. import config


def _gen(fn):
    '''Global generator for all formats'''
    return (line for line in fn if 'NEW DATA' not in line)


def _open(*args, **kwargs):
    if kwargs.get('bz2', False):
        return _bz2.open(args[0], 'rt')
    else:
        return open(*args)


def _get(_it, kernel, **kwargs):
    '''Gets batch of lines defined by _n_lines and processes
       it with given _kernel. Returns processed data.'''

    n_lines = kwargs.get('n_lines')

    _range = kwargs.pop("range", (0, 1, float('inf')))
    if len(_range) == 2:
        r0, r1 = _range
        _ir = 1
    elif len(_range) == 3:
        r0, _ir, r1 = _range
    else:
        raise ValueError('Given range is not a tuple of length 2 or 3!')

    _sk = kwargs.pop("skip", [])

    class _line_iterator():
        '''self._r ... the frame that will be returned next (!)'''
        def __init__(self):
            self.current_line = 0
            self._it = _it
            self._r = 0
            self._skip = _sk
            self._offset = 0
            while self._r < r0:
                [next(_it) for _ik in range(n_lines)]
                if self._r + self._offset in _sk:
                    self._skip.remove(self._r + self._offset)
                    self._offset += 1
                else:
                    self._r += 1

        def __iter__(self):
            return self

        def __next__(self):
            while (self._r - r0) % _ir != 0 or self._r + self._offset in _sk:
                [next(_it) for _ik in range(n_lines)]
                if self._r + self._offset in _sk:
                    self._skip.remove(self._r + self._offset)
                    self._offset += 1
                else:
                    self._r += 1

            self._r += 1
            if r1 > 0 and self._r > r1:
                raise StopIteration()

            return islice(_it, n_lines)

    _data = _line_iterator()
    while True:
        try:
            yield kernel(next(_data), **kwargs)
        except StopIteration:
            break


def _reader(FN, n_lines, kernel,
            convert=1,
            verbose=config.__verbose__,
            bz2=False,
            **kwargs):
    '''Opens file, checks contents, and parses arguments,
       kernel, and generator.'''

    with _open(FN, 'r', bz2=bz2, **kwargs) as _f:
        _it = _gen(_f)
        data = tqdm(_get(_it,
                         kernel,
                         convert=convert,
                         n_lines=n_lines,
                         **kwargs),
                    desc="%30s" % FN,
                    disable=not verbose)

        if np.size(data) == 0:
            raise ValueError('Given input and arguments '
                             'do not yield any data!')
        else:
            for _d in data:
                yield _d


def _dummy_kernel(frame, **kwargs):
    '''Simplest _kernel. Does nothing.'''
    return frame
