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


from itertools import islice
import numpy as np


def _gen(fn):
    '''Global generator for all formats'''
    return (line for line in fn if 'NEW DATA' not in line)


def _get(_it, kernel, **kwargs):
    '''Gets batch of lines defined by _n_lines and processes
       it with given _kernel. Returns processed data.'''

    n_lines = kwargs.get('n_lines')
    r0, _ir, r1 = kwargs.pop("range", (0, 1, float('inf')))
    _sk = kwargs.get("skip", [])

    class _line_iterator():
        '''self._r ... the frame that will be returned next (!)'''
        def __init__(self):
            self.current_line = 0
            self._it = _it
            self._r = 0
            while self._r < r0:
                [next(_it) for _ik in range(n_lines)]
                self._r += 1

        def __iter__(self):
            return self

        def __next__(self):
            while (self._r - r0) % _ir != 0 or self._r in _sk:
                [next(_it) for _ik in range(n_lines)]
                self._r += 1

            self._r += 1
            if self._r > r1:
                raise StopIteration()

            return islice(_it, n_lines)

    _data = _line_iterator()
    while True:
        try:
            yield kernel(next(_data), **kwargs)
        except StopIteration:
            break


def _reader(FN, _nlines, _kernel, **kwargs):
    '''Opens file, checks contents, and parses arguments,
       _kernel, and generator.'''

    kwargs.update({'n_lines': _nlines})

    with open(FN, 'r') as _f:
        _it = _gen(_f)
        data = _get(_it, _kernel, **kwargs)

        if np.size(data) == 0:
            raise ValueError('Given input and arguments '
                             'do not yield any data!')
        else:
            for _d in data:
                    yield _d


def _dummy_kernel(frame, **kwargs):
    '''Simplest _kernel. Does nothing.'''
    return frame
