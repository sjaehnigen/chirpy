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


class TestPhysics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestInterfaces(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    # orca, molden, vmd, pymol, ...


# Rather than checking the next two, check if they still have methods that should not be in class
# Test core classes

class TestClasses(unittest.TestCase):
    # Later for trajectory class: load vels, find unknown elements

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestCreate(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


# Test the binaries (all arguments): for i in *py; do ./$i -h 1>/dev/null; done
# --- not tested
#       visualise, MDAnalysis, test

