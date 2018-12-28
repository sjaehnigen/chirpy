#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import imp

from physics import constants
from interfaces import cpmd
from topology import dissection
from classes import molecule


def main():
    '''Split CPMD job based on existing input file ( trajectory optional )'''
    parser = argparse.ArgumentParser( description = "Split CPMD job based on existing input file ( trajectory optional )", 
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter
                                    )
    parser.add_argument( "fn", help = "Existing cpmd input file" )
    parser.add_argument( "-t", help = "Existing cpmd trajectory file", default = None )
    args = parser.parse_args( )


    #_dir = '/home/ThC/TMP/'
    #fn_inp = _dir + 'cpmd_tcd.inp'
    #fn_trj = _dir + 'TRAJECTORY'
    fn_inp = args.fn
    fn_trj = args.t
    
    # Read CPMD input an adapt options
    _JOB = cpmd.CPMDjob.read_input_file( fn_inp )
    _JOB.CPMD.options[ "CENTER MOLECULE" ] = ( [ 'OFF' ], None )
    
    if fn_trj is not None:
        _JOB.TRAJECTORY = cpmd.TRAJECTORY.read( fn_trj, sum( _JOB.ATOMS.n_kinds ), symbols = _JOB.get_symbols( ) )
    
    # Get Fragments
    _XYZ = molecule.XYZData( data = _JOB.get_positions(), symbols = _JOB.get_symbols() )
    _ASS = dissection.define_molecules_XYZclass( _XYZ )
    print ( _ASS )
    
    # Write fragment output
    for _i, _FRAG in enumerate( _JOB.split_atoms( _ASS ) ):
        _FRAG.write_input_file( fn_inp[ : -4 ] + '_fragment_%03d' % _i + '.inp' )
        if fn_trj is not None:
            _FRAG.TRAJECTORY.write( fn_trj + '_fragment_%03d' % _i, fmt = 'cpmd' )
    
if(__name__ == "__main__"):
    main()
