#!/usr/bin/env python

def extract_keys(dict1, **defaults):
    '''Updates the keys/value pairs of dict1 with those of dict2.
    Similar to dict1.update(dict2), but it does not ADD any new keys to dict1.'''
    return {_s : dict1.get(_s, defaults[_s]) for _s in defaults}
