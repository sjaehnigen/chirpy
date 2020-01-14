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

import numpy as _np
import copy as _copy

from ..interface import cpmd

# DEV log
# Todo:
#   implement TRAJECTORY
#   initialise with a default cpmd job (SCF) ==> where to put defaults?
#   check for mandatory sections ?
#   just pass **kwargs to respective section init (via nargs)
# END


class CPMDjob():
    def __init__(self, **kwargs):
        # mandatory
        for _sec in [
                'INFO',
                'CPMD',
                'SYSTEM',
                'DFT',
                'ATOMS',
                ]:
            setattr(self, _sec, kwargs.get(_sec, getattr(cpmd, _sec)()))
        self.INFO = cpmd.INFO(options={'CPMD DEFAULT JOB': ([], None)})

        # optional
        for _sec in [
                'RESP',
                # 'TRAJECTORY',
                ]:
            setattr(self, _sec, kwargs.get(_sec, getattr(cpmd, _sec)()))

        self._check_consistency()

    def _check_consistency(self):
        pass

    @classmethod
    def read_input_file(cls, fn):
        '''CPMD 4'''

        def _parse_file(_iter):
            _l = next(_iter)
            _c = []
            while "&END" not in _l:
                _c.append(_l)
                _l = next(_iter)
            return iter(_c)

        with open(fn, 'r') as _f:
            _iter = (_l.strip() for _l in _f)
            CONTENT = {_l[1:].upper(): _parse_file(_iter)
                       for _l in _iter if "&" in _l}

        CONTENT = {_C: getattr(cpmd, _C)._parse_section_input(CONTENT[_C])
                   for _C in CONTENT}

        return cls(**CONTENT)

    def write_input_file(self, *args):
        ''' CPMD 4 '''

        # known sections and order
        _SEC = ['INFO', 'CPMD', 'RESP', 'DFT', 'SYSTEM', 'ATOMS']

        for _s in _SEC:
            if hasattr(self, _s):
                getattr(self, _s).write_section(*args)

    def get_positions(self):
        ''' in a. u. ? '''
        return _np.vstack(self.ATOMS.data)

    def get_symbols(self):
        symbols = []
        for _s, _n in zip(self.ATOMS.kinds, self.ATOMS.n_kinds):
            symbols += _s.split('_')[0][1:] * _n
        return _np.array(symbols)

    def get_kinds(self):
        kinds = []
        for _s, _n in zip(self.ATOMS.kinds, self.ATOMS.n_kinds):
            kinds += [_s] * _n
        return _np.array(kinds)

    def get_channels(self):
        channels = []
        for _s, _n in zip(self.ATOMS.channels, self.ATOMS.n_kinds):
            channels += [_s] * _n
        return _np.array(channels)

    def split_atoms(self, ids):
        ''' Split atomic system into fragments and create CPMD job,
            respectively.
            ids ... list  of fragment id per atom
            (can be anything str, int, float, ...).
            '''
        def _dec(_P):
            return [[_P[_k] for _k, _jd in enumerate(ids) if _jd == _id]
                    for _id in sorted(set(ids))]
        # problem: set is UNSORTED ==> see TRAJECTORY._sort routine
        # for another way

        _L = []

        if hasattr(self, 'TRAJECTORY'):
            _getlist = [
                        self.get_positions(),
                        self.get_kinds(),
                        self.get_channels(),
                        self.TRAJECTORY.pos_aa.swapaxes(0, 1),
                        self.TRAJECTORY.vel_au.swapaxes(0, 1),
                      ]
        else:
            _getlist = [
                        self.get_positions(),
                        self.get_kinds(),
                        self.get_channels(),
                       ]

#        for _sd, _sk, _sch in zip(*map(_dec, _getlist)):
        for _frag in zip(*map(_dec, _getlist)):
            _C = {}
            _C['kinds'] = []
            _C['channels'] = []
            _C['n_kinds'] = []
            _C['data'] = []
            # complicated to keep order
            _isk_old = None
            # for _I, _isk in enumerate(sorted(set(_frag[1]))):
            for _isk in _frag[1]:
                if _isk != _isk_old:
                    _C['kinds'].append(_isk)
                    _C['channels'].append(
                            [_frag[2][_j]
                             for _j, _jsk in enumerate(_frag[1])
                             if _jsk == _isk
                             ][0])  # a little awkward
                    _C['n_kinds'].append(_frag[1].count(_isk))
                    _C['data'].append(
                            _np.array([_frag[0][_j]
                                      for _j, _jsk in enumerate(_frag[1])
                                      if _jsk == _isk])
                            )
                _isk_old = _copy.deepcopy(_isk)  # necessary?

            out = _copy.deepcopy(self)
            out.ATOMS = cpmd.ATOMS(**_C)

            if hasattr(self, 'TRAJECTORY'):
                out.TRAJECTORY = cpmd.TRAJECTORY(
                                     pos_aa=_np.array(_frag[3]).swapaxes(0, 1),
                                     vel_au=_np.array(_frag[4]).swapaxes(0, 1),
                                     symbols=out.get_symbols(),
                                                 )

            _L.append(out)

        return _L
