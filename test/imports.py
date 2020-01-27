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

import unittest
import importlib
import pkgutil


class TestImports(unittest.TestCase):

    @staticmethod
    def import_submodules(package, recursive=True):
        '''https://stackoverflow.com/a/25562415'''
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + '.' + name
            results[full_name] = importlib.import_module(full_name)
            if recursive and is_pkg:
                results.update(TestImports.import_submodules(full_name))
        return results

    def test_import_00_read(self):
        self.import_submodules('chirpy.read')

    def test_import_00_write(self):
        self.import_submodules('chirpy.write')

    def test_import_00_topology(self):
        self.import_submodules('chirpy.topology')

    def test_import_00_mathematics(self):
        self.import_submodules('chirpy.mathematics')

    def test_import_00_physics(self):
        self.import_submodules('chirpy.physics')

    def test_import_01_interface(self):
        # numpy warning for scipy.interpolate import UnivariateSpline in vmd
        self.import_submodules('chirpy.interface')

    def test_import_01_mdanalysis(self):
        self.import_submodules('chirpy.mdanalysis')

    def test_import_02_classes(self):
        self.import_submodules('chirpy.classes')

    def test_import_03_create(self):
        self.import_submodules('chirpy.create')

    def test_import_10_visualise(self):
        self.import_submodules('chirpy.visualise')

    def test_import_10_external(self):
        self.import_submodules('chirpy.external')
