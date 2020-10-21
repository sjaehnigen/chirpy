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
import numpy as np
import multiprocessing
from multiprocessing import Pool
from itertools import product, combinations_with_replacement


class AttrDict(dict):
    '''Converts dictionary keys into attributes'''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # --- protect built-in namespace of dict
        for _k in self:
            if _k in dir(self):
                raise NameError("Trying to alter namespace of built-in "
                                "dict method!", _k)
        self.__dict__ = self

    def __setitem__(self, key, value):
        if key in dir(self):
            raise NameError("Trying to alter namespace of built-in "
                            "dict method!", key)
        super(AttrDict, self).__setitem__(key, value)


class _PALARRAY():
    '''Class for parallel processing of array data.
       Processes data in a nested grid
       (i.e. all combinations of indices)'''

    def __init__(self, func, *data, repeat=1,
                 n_cores=multiprocessing.cpu_count(),
                 upper_triangle=False,
                 axis=0):
        '''Parallel array process setup.
           With executable func(*args) and set of data arrays, data,
           whereas len(data) = len(args).

           Calculate upper_triangle matrix only, if len(data)==1 and repeat==2
           (optional).

           Output: array of shape (len(data[0], len(data[1]), ...)'''

        self.f = func
        self.pool = Pool(n_cores)

        self.data = tuple([np.moveaxis(_d, axis, 0) for _d in data])
        self._ut = False
        self.repeat = repeat

        # ToDo: memory warning:
        # print(np.prod([len(_d) for _d in self.data]))

        if upper_triangle:
            self._ut = True
            if len(data) == 1 and repeat == 2:
                self.array = combinations_with_replacement(self.data[0], 2)
            else:
                raise ValueError('Upper triangle requires one data array with '
                                 'repeat=2!')
        else:
            self.array = product(*self.data, repeat=repeat)

    def run(self):
        try:
            result = np.array(self.pool.starmap(self.f, self.array))

            _l = self.repeat * tuple([len(_d) for _d in self.data])
            if self._ut:
                res_result = np.zeros(_l + result.shape[1:])
                res_result[np.triu_indices(_l[0])] = result

                return res_result

            else:
                return result.reshape(_l + result.shape[1:])

        except KeyboardInterrupt:
            print("KeyboardInterrupt in _PALARRAY")

        finally:
            self.pool.terminate()
            self.pool.join()


class _CORE():
    def __init__(self, *args, **kwargs):
        self._print_info = []

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self = self.__add__(other)
        return self

    def __rsub__(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        self = self.__sub__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self = self.__mul__(other)
        return self

    def __ipow__(self, other):
        self = self.__pow__(other)
        return self

    def dump(self, FN):
        with open(FN, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, FN):
        with open(FN, "rb") as f:
            _load = pickle.load(f)
        if not isinstance(_load, cls):
            warnings.warn(f"{FN} does not contain {cls.__name__}. Trying to "
                          "convert.", stacklevel=2)
            _load = convert_object(_load, cls)

        # if hasattr(_load, 'data'):
            # warnings.warn(f"Found data with shape {_load.data.shape}.",
            #               stacklevel=2)

        if hasattr(_load, '_sync_class'):
            _load._sync_class()

        return _load

    def print_info(self):
        print('')
        print(77 * '–')
        print('%-12s' % self.__class__.__name__)
        print(77 * '–')
        print('')

        for _func in self.print_info:
            _func(self)


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
        # --- Store original skip information as it is consumed by generator
        self._kwargs['_skip'] = self._kwargs['skip'].copy()

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, '_chaste'):
            # --- repeat first step of next() after __init__
            # --- do nothing
            if self._chaste:
                # warnings.warn("Currently loaded frame repeated after init.",
                #              stacklevel=2)
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
        '''Reinitialises the iterator'''
        self._chaste = False
        if '_skip' in self._kwargs:
            self._kwargs['skip'] = self._kwargs['_skip'].copy()
        self.__init__(self._fn, **self._kwargs)

    def _unwind(self, *args, **kwargs):
        '''Unwinds the Iterator according to <length> or until it is exhausted
           constantly executing the given frame-owned function and passing
           through given arguments.
           Events are dictionaries with (relative) frames
           as keys and some action as argument that are only
           executed when the Iterator reaches the value of the
           key.
           (This can partially also be done with masks.)
           '''
        func = kwargs.pop('func', None)
        events = kwargs.pop('events', {})
        length = kwargs.pop('length', None)
        _fr = 0
        for _ifr in itertools.islice(self, length):
            kwargs['frame'] = _ifr
            if isinstance(func, str):
                getattr(self._frame, func)(*args, **kwargs)
            elif callable(func):
                func(self, *args, **kwargs)
            if _fr in events:
                if isinstance(events[_fr], dict):
                    kwargs.update(events[_fr])
            _fr += 1

        if length is not None:
            next(self)
            self._chaste = True

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

    def mask_duplicate_frames(self, verbose=True, **kwargs):
        # ToDo: Generalise this function for all kinds of ITERATORS
        def split_comment(comment):
            if 'i = ' in comment:
                return int(comment.split()[2].rstrip(','))
            elif 'Iteration:' in comment:
                return int(comment.split('_')[1].rstrip())
            else:
                raise TypeError('Cannot get frame info from comments!')

        def _func(obj, **kwargs):
            _skip = obj._kwargs.get('skip', [])
            _timesteps = obj._kwargs.get('_timesteps', [])
            _ts = split_comment(obj._frame.comments)
            if _ts not in _timesteps:
                _timesteps.append(_ts)
            else:
                if verbose:
                    print(obj._fr, ' doublet of ', _ts)
                _skip.append(obj._fr)
            obj._kwargs.update({'_timesteps': _timesteps})
            obj._kwargs.update({'skip': _skip})

        _keep = self._kwargs['range']
        _masks = self._kwargs['_masks']

        if self._kwargs['range'][1] != 1:
            warnings.warn('Setting range increment to 1 for doublet search!',
                          stacklevel=2)
            self._kwargs['range'] = (_keep[0], 1, _keep[2])
            self.rewind()

        if len(self._kwargs['_masks']) > 0:
            warnings.warn('Disabling masks for doublet search! %s' % _masks,
                          stacklevel=2)
            self._kwargs['_masks'] = []

        self._kwargs['_timesteps'] = []
        kwargs['func'] = _func
        try:
            self._unwind(**kwargs)
        except ValueError:
            raise ValueError('Broken trajectory! Stopped at frame %s (%s)'
                             % (self._fr, self.comments))

        # ToDo: re-add this warning without using numpy
        if len(np.unique(np.diff(self._kwargs['_timesteps']))) != 1:
            warnings.warn("CRITICAL: Found varying timesteps!", stacklevel=2)

        if verbose:
            print('Duplicate frames in %s according to range %s:' % (
                    self._fn,
                    self._kwargs['range']
                    ), self._kwargs['skip'])

        self._kwargs['_masks'] = _masks
        self._kwargs['range'] = _keep

        # --- Store original skip information as it is consumed by generator
        self._kwargs['_skip'] = self._kwargs['skip'].copy()

        if kwargs.get('rewind', True):
            self.rewind()

        return self._kwargs['skip']


# BETA: Work in progress

_known_objects = {
        'VectorField': {
            'VectorField': [],
            'CurrentDensity': [],
            'TDElectronDensity': ['j'],
            'TDElectronicState': ['j'],
            },
        'ScalarField': {
            'ScalarField': [],
            'ElectronDensity': [],
            'WaveFunction': [],
            'WannierFunction': [],
            'TDElectronDensity': ['rho'],
            'TDElectronicState': ['psi'],
            # 'TDElectronicState': ['psi1'],
            }
        }

_attributes = {
        'VectorField': [
            'data',
            'cell_vec_au',
            'origin_au',
            'pos_au',
            'numbers',
            'comments'
            ],
        'ScalarField': [
            'data',
            'cell_vec_au',
            'origin_au',
            'pos_au',
            'numbers',
            'comments'
            ],
        }


def convert_object(source, target):
    '''source is an object and target is a class'''
    try:
        _ko = _known_objects[target.__name__]

    except KeyError:
        raise TypeError('Unsupported conversion target: '
                        f'{target.__name__}')

    src = source
    src_name = source.__class__.__name__
    if src_name in _ko:
        for attr in _ko[src_name]:
            src = getattr(src, attr)
    else:
        raise TypeError(f'{target.__name__} cannot proselytize to '
                        '{source.__class__.__name__}!')

    obj = target.__new__(target)
    for attr in _attributes[target.__name__]:
        value = getattr(src, attr)
        if value is None:
            raise AttributeError(f'{source} does not contain the {attr} '
                                 'attribute!')
        setattr(obj, attr, value)

    return obj