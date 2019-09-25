#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


#For basic python functionalities 
def extract_keys(dict1, **defaults):
    '''Updates the keys/value pairs of defaults with those of dict1.
    Similar to defaults.update(dict1), but it does not ADD any new keys to dict1.'''
    return {_s : dict1.get(_s, defaults[_s]) for _s in defaults}
