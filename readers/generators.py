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
    _r = 0

    # while _r < r0:
    #     [next(_it) for _ik in range(n_lines)]
    #     _r += 1

    class _line_iterator():
        def __init__(self):
            self.current_line = 0
            self._it = _it
            self._r = 0
            try:
                while _r < r0:
                    [next(_it) for _ik in range(n_lines)]
                    self._r += 1
            except StopIteration:
                print("ENDE in init")

        def __iter__(self):
            return self
        def __next__(self):
            while self._r % _ir != 0:
                [next(_it) for _ik in range(n_lines)]
                self._r += 1
 #           self.current_line += 1
 #           if self.current_line < n_lines:
            # if self._r > 0: print(next(_it))
            self._r += 1
            # return (next(_it) for _ik in range(n_lines))
            # return (_iit for _iit, _ik in izip(_it, range(n_lines)))
            return islice(_it, n_lines)

    _data = _line_iterator()
    while True:
        #_data = None
        try:
        # while True:
            # def _data():
            #     for _ik in range(n_lines):
            #         _l = next(_it)
            #         if _r % _ir == 0:
            #             yield _l
            #             # _data.append(_l)
            # _data = (_l for _l in islice(_it, n_lines) if _r % _ir == 0)
            # if _r % _ir == 0:
            #    _data = islice(_it, n_lines)
            #if _r % _ir == 0:
            yield kernel(next(_data), **kwargs)
            #else:
            #    trash = tuple(_data)
            #    print(trash.shape)
            #_r += 1
            #if _r >= r1:
            #    _data = None
            #    raise StopIteration()

        except StopIteration:
            # if _data is not None:  # len(_data) != 0:
            #    raise ValueError('Reached end of while processing frame!')
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
