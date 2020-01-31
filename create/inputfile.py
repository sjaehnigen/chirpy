#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------

from ..interface import cpmd


class QMCalculation():
    def __init__(self, *args, **kwargs):
        self.functional = None
        self.type = 'MD sampling'
        self.eps = 1.E-7

    def write_input_file(self, fn, code='cpmd'):
        if code == 'cpmd':
            cpmd.CPMDjob().write_input_file(fn)
