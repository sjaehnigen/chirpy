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

import pickle


class _CORE():

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
