# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2023, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------

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

    def test_import_02_classes(self):
        self.import_submodules('chirpy.classes')

    def test_import_03_create(self):
        self.import_submodules('chirpy.create')

    def test_import_10_visualise(self):
        self.import_submodules('chirpy.visualise')
