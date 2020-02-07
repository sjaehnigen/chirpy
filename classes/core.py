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

import pickle
import itertools
import warnings


class _CORE():
    def __init__(self, *args, **kwargs):
        self.data = kwargs.get("data")
        self._sync_class()

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self = self.__mul__(other)
        return self

    def __ipow__(self, other):
        self = self.__pow__(other)
        return self

    def _sync_class(self):
        self.n_fields = self.data.shape[-1]

    def dump(self, FN):
        with open(FN, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, FN):
        with open(FN, "rb") as f:
            _load = pickle.load(f)
        if isinstance(_load, cls):
            return _load
        else:
            raise TypeError("File does not contain %s." % cls.__name__)

    def print_info(self):
        print('')
        print(77 * '–')
        print('%-12s' % self.__class__.__name__)
        print(77 * '–')
        print('')


class _ITERATOR():
    def __init__(self, *args, **kwargs):
        self._kernel = _CORE
        self._gen = iter([])
        self._kwargs = {}
        # --- initialise list of masks
        self._kwargs['_masks'] = []
        # --- keep kwargs for iterations
        self._kwargs.update(kwargs)

        # --- Load first frame w/o consuming it
        self._fr -= self._st
        next(self)
        self._chaste = True

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, '_chaste'):
            # --- repeat first step of next() after __init__
            # --- do nothing
            if self._chaste:
                self._chaste = False
                return self._fr

        frame = next(self._gen)

        out = {'data': frame}

        self._fr += self._st
        self._kwargs.update(out)

        self._frame = self._kernel(**self._kwargs)

        # --- check for stored masks
        for _f, _f_args, _f_kwargs in self._kwargs['_masks']:
            if isinstance(_f, str):
                getattr(self._frame, _f)(*_f_args, **_f_kwargs)
            elif callable(_f):
                self._frame = _f(self._frame, *_f_args, **_f_kwargs)

        self.__dict__.update(self._frame.__dict__)

        return self._fr

    @classmethod
    def _from_list(cls, LIST, **kwargs):
        a = cls(LIST[0], **kwargs)
        for _f in LIST[1:]:
            b = cls(_f, **kwargs)
            a += b
        return a

    def __add__(self, other):
        new = self
        if new._topology._is_similar(other._topology)[0] == 1:
            new._gen = itertools.chain(new._gen, other._gen)
            return new
        else:
            raise ValueError('Cannot combine frames of different size!')

    def rewind(self):
        '''Reinitialises the Interator (BETA)'''
        self._chaste = False
        self.__init__(self._fn, **self._kwargs)

    def _unwind(self, *args, **kwargs):
        '''Unwinds the Iterator until it is exhausted constantly
           executing the given frame-owned function and passing
           through given arguments.
           Events are dictionaries with (relative) frames
           as keys and some action as argument that are only
           executed when the Iterator reaches the value of the
           key.
           This can partially also be done with masks.'''
        func = kwargs.pop('func', None)
        events = kwargs.pop('events', {})
        _fr = 0
        for _ifr in self:
            kwargs['frame'] = _ifr
            if isinstance(func, str):
                getattr(self._frame, func)(*args, **kwargs)
            elif callable(func):
                func(self, *args, **kwargs)
            if _fr in events:
                if isinstance(events[_fr], dict):
                    kwargs.update(events[_fr])
            _fr += 1

    @staticmethod
    def _mask(obj, func, *args, **kwargs):
        '''Adds a frame-owned function that is called with every __next__()
           before returning.'''
        obj._kwargs['_masks'].append(
                (func, args, kwargs),
                )
        if len(obj._kwargs['_masks']) > 10:
            warnings.warn('Too many masks on iterator!', stacklevel=2)

    def merge(self, other, **kwargs):
        '''Merge with other object by combining the two iterators other than
           along principal axis (use "+") for that.
           Specify axis 0 or 1 to combine atoms or data, respectively
           (default: 0).
           Other iterator should not be used anymore!
           BETA'''

        def _func(obj1, obj2, **kwargs):
            # --- next(obj1) is called before loading mask
            try:
                next(obj2)
                obj1 += obj2
                return obj1
            except StopIteration:
                with warnings.catch_warnings():
                    warnings.warn('Merged iterator exhausted!',
                                  RuntimeWarning,
                                  stacklevel=1)
                return obj1

        self._frame += other._frame
        self.__dict__.update(self._frame.__dict__)
        other._chaste = False

        self._mask(self, _func, other, **kwargs)
