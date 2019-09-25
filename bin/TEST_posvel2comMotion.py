#!/usr/bin/env python
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


import numpy as np
import argparse

from chirpy.physics import constants #migrate to phdtools

from fileio import xyz #migrate to phdtools

def main():
    parser=argparse.ArgumentParser(description="TEST_posvel2comMotion.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fn_inp',                                         help="posvel input file (e.g., xyz file)")
    #parser.add_argument("--verbose",  action='store_true', default=False, help="Verbose output")
    #add atom-wise info?
    args = parser.parse_args()
    fn_inp = args.fn_inp
    fmt = fn_inp.split('.')[-1]

    if fmt=="xyz":
        data, symbols, comments = xyz.ReadTrajectory_BruteForce(fn_inp)
#    if fmt=="xvibs":
#        self.fn = fn
#        comments = ["xvibs"]
#        n_atoms, numbers, coords_aa, n_modes, omega_invcm, modes = xvibs.ReadXVIBFile(fn)
#        symbols  = [constants.symbols[z-1] for z in numbers]
#        data     = coords_aa.reshape((1,n_atoms,3))
    else:
        raise Exception('Unknown format: %s.'%fmt)

    try:
        masses = np.array([constants.species[s]['MASS'] for s in symbols])
        print('COM motion is\n%s'%(1/np.sum(masses)*(data[:,:,3:]*masses[None,:,None]).sum(axis=1)))
        print('COG motion is\n%s'%data[:,:,3:].sum(axis=1))

#        #frame 0
#        angmoms = np.zeros(data[0,0,3:].shape)
#        for i, mass in enumerate(masses):
#            angmoms += np.cross(data[0,i,:3],data[0,i,3:])*mass
#        print(angmoms)

    except IndexError:
        raise Exception('Input file does not have the correct shape (n_frames,n_atoms,posvel)')


if (__name__ == '__main__'):
    main()

