#!/usr/bin/env python
# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
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
# -------------------------------------------------------------------

import unittest
import sys

from chirpy.version import __version__
from chirpy import config

import imports
import read
import write
import interface
import mathematics
import topology
import physics
import classes
import create
import bin

if __name__ == '__main__':
    try:
        _verbosity = int(sys.argv[1])
    except IndexError:
        _verbosity = 1
    config.set_verbose(False)
    if _verbosity > 2:
        config.set_verbose(True)

    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))

    print(f'You are using ChirPy version {__version__}')
    print(70 * '-')
    print('Running TestSuite')
    sys.stdout.flush()

    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add tests to the test suite
    suite.addTests(loader.loadTestsFromModule(imports))
    suite.addTests(loader.loadTestsFromModule(read))
    suite.addTests(loader.loadTestsFromModule(write))
    suite.addTests(loader.loadTestsFromModule(interface))
    suite.addTests(loader.loadTestsFromModule(mathematics))
    suite.addTests(loader.loadTestsFromModule(topology))
    suite.addTests(loader.loadTestsFromModule(physics))
    suite.addTests(loader.loadTestsFromModule(classes))
    suite.addTests(loader.loadTestsFromModule(create))
    suite.addTests(loader.loadTestsFromModule(bin))

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=_verbosity)

    result = runner.run(suite)
