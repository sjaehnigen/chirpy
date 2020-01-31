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

from chirpy.test import \
    imports,\
    read,\
    write,\
    interface,\
    mathematics,\
    topology,\
    physics,\
    classes,\
    create

# --- INSUFFICIENT tests for
# physics
# create
# classes

# --- NOT (yet) tested
# snippets
# bin
# external
# visualise


if __name__ == '__main__':
    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))

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

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=1)

    result = runner.run(suite)
