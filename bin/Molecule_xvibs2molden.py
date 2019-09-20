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

import numpy as np
from chirpy.readers.modes import xvibsReader
from chirpy.interfaces import molden
from chirpy.physics import constants
import argparse
import os

#by-pass Molecule class for now
if(__name__ == "__main__"):
    print('WARNING: BETA version!')
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_inp", help="Input xvibs-File")
    parser.add_argument("-f", help="Output file name (standard: 'output.mol')", default='output.mol')
    parser.add_argument("-mw", action="store_true",help="modes in xvibs file denote mass weighted displacements (default: False --> actual cartesian displacements)", default=False)
    args = parser.parse_args()
    if not os.path.exists(args.fn_inp):
        raise Exception('File %s does not exist' % args.fn_inp)

    n_atoms, numbers, coords_aa, n_modes, freqs, modes = xvibsReader(args.fn_inp)
    symbols  = [constants.symbols[z-1] for z in numbers]
    masses   = [constants.species[s]['MASS'] for s in symbols]
    coords_au = coords_aa*constants.l_aa2au
    if args.mw:
        print('Assuming mass-weighted coordinates in XVIBS.')
        modes /= np.sqrt(masses)[None,:,None]
    else:
        print('Not assuming mass-weighted coordinates in XVIBS.')
    modes = modes.reshape((n_modes,n_atoms*3)) 
    print(modes.shape)
    freqs = np.array(freqs)
    molden.WriteMoldenVibFile(args.f, symbols, coords_au, freqs, modes)


