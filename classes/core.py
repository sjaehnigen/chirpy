#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------
#!/usr/bin/env python
#Version important as <3.6 gives problems with OrderedDictionaries

import sys
import os
import copy
import numpy as np

from ..writers.trajectory import cpmdWriter, xyzWriter, pdbWriter
from ..physics import constants

#put this into new lib file
valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
Angstrom2Bohr = 1.8897261247828971
np.set_printoptions(precision=5,suppress=True)

#not so clear how to sort all this out
#distant goal: python-based setup for any MD simulation

class Project( ):
    pass
    #this versatile class shall collect all parametres

