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

from ..interface import cpmd


class QMCalculation():
    def __init__(self, *args, **kwargs):
        self.functional = None
        self.type = 'MD sampling'
        self.eps = 1.E-7

    def write_input_file(self, fn, code='cpmd'):
        if code == 'cpmd':
            cpmd.CPMDjob().write_input_file(fn)
