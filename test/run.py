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

# import os
import unittest

# import your test modules
# import import
from chirpy.test import \
    imports,\
    read,\
    write,\
    interface,\
    mathematics,\
    topology  # ,\
    # physics,\
    # classes,\
    # create,\
    # visualise,\

# --- ToDo (change import path later after moving bin)
# import mdanalysis import
# read iterators

# --- NOT (yet) tested
# snippets
# bin
# external
# visualisation


if __name__ == '__main__':
    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))

    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add tests to the test suite
    # suite.addTests(loader.loadTestsFromModule(imports))
    # suite.addTests(loader.loadTestsFromModule(read))
    # suite.addTests(loader.loadTestsFromModule(write))
    # suite.addTests(loader.loadTestsFromModule(interface))
    suite.addTests(loader.loadTestsFromModule(mathematics))
    # suite.addTests(loader.loadTestsFromModule(topology))

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=4)

    result = runner.run(suite)
