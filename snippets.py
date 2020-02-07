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

import warnings
import numpy as _np


def extract_keys(dict1, **defaults):
    '''Updates the key/value pairs of defaults with those of dict1.
       Similar to defaults.update(dict1), but it does not ADD any new keys to
       defaults.'''
    return {_s: dict1.get(_s, defaults[_s]) for _s in defaults}


def tracked_update(dict1, dict2):
    '''Update dict1 with dict2 but warn if existing data is changed'''
    for _k2 in dict2:
        # python3.8: use walrus
        _v1 = dict1.get(_k2)
        _v2 = dict2.get(_k2)
        if _v1 is not None:
            if not equal(_v1, _v2):
                with warnings.catch_warnings():
                    warnings.warn('Overwriting existing key \'{}\' in '
                                  'dictionary!'.format(_k2),
                                  RuntimeWarning,
                                  stacklevel=2)
    dict1.update(dict2)

    return dict1


def equal(a, b):
    '''return all-equal regardless of type'''
    if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
        return _np.all(a == b)
    else:
        return a == b
