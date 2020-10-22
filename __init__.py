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

import sys
import warnings
import numpy as _np

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")


def extract_keys(dict1, **defaults):
    '''Updates the key/value pairs of defaults with those of dict1.
       Similar to defaults.update(dict1), but it does not ADD any new keys to
       defaults.'''
    return {_s: dict1.get(_s, defaults[_s]) for _s in defaults}


def tracked_extract_keys(dict1, **defaults):
    '''Updates the key/value pairs of defaults with those of dict1.
       Similar to defaults.update(dict1), but it does not ADD any new keys to
       defaults.
       Warns if existing data is changed.'''
    msg = defaults.pop('msg', 'in dict1!')
    new_dict = {_s: dict1.get(_s, defaults[_s]) for _s in defaults}

    return tracked_update(defaults, new_dict, msg=msg)


def tracked_update(dict1, dict2, msg='in dict1!'):
    '''Update dict1 with dict2 but warns if existing data is changed'''
    for _k2 in dict2:
        _v1 = dict1.get(_k2)
        _v2 = dict2.get(_k2)
        if _v1 is not None:
            if not equal(_v1, _v2):
                with warnings.catch_warnings():
                    warnings.warn('Overwriting existing key '
                                  '\'{}\' '.format(_k2) + msg,
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
