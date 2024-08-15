#!/usr/bin/env python
# ----------------------------------------------------------------------
#
#  ChirPy
#
#    A python package for chirality, dynamics, and molecular vibrations.
#
#    https://github.com/sjaehnigen/chirpy
#
#
#  Copyright (c) 2020-2024, The ChirPy Developers.
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

import argparse
import unittest
import sys

import chirpy as cp

import imports
import read
import write
import interface
import mathematics
import topology
import physics
import classes
import create
import scripts

sys.tracebacklimit = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Run TestSuite",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            default=False,
            )
    parser.add_argument(
            "--scripts",
            action="store_true",
            help="Test chirpy scripts. "
                 "The script directory needs to be added to PATH",
            default=False,
            )
    args = parser.parse_args()

    _verbosity = 1
    # --- let the TestSuite speak
    cp.config.set_verbose(False)
    if args.verbose:
        _verbosity = 2
        cp.config.set_verbose(True)

    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))

    print(f'You are using ChirPy version {cp.__version__} on '
          f'{cp.config.__os__}')
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
    if args.scripts:
        suite.addTests(loader.loadTestsFromModule(scripts))

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=_verbosity)

    result = runner.run(suite)
