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
    create,\
    bin

# --- INSUFFICIENT tests for
# physics
# create
# classes
# bin

# --- NOT (yet) tested
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
    suite.addTests(loader.loadTestsFromModule(bin))

    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=1)

    result = runner.run(suite)
