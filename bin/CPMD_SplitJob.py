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


import argparse

from chirpy.interfaces import cpmd
from chirpy.topology import dissection
from chirpy.classes import system


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
    _JOB.CPMD.options.pop( "RESTART COORDINATES WAVEFUNCTION" )  #keyword logic not yet in cpmd class

    if fn_trj is not None:
        _JOB.TRAJECTORY = cpmd.TRAJECTORY.read( fn_trj, sum( _JOB.ATOMS.n_kinds ), symbols = _JOB.get_symbols( ) )

    # Get Fragments
    _XYZ = system.XYZData( data = _JOB.get_positions(), symbols = _JOB.get_symbols() )
    _ASS = dissection.define_molecules_XYZclass( _XYZ )
    print ( _ASS )

    # Write fragment output
    for _i, _FRAG in enumerate( _JOB.split_atoms( _ASS ) ):
        _FRAG.write_input_file( fn_inp[ : -4 ] + '_fragment_%03d' % _i + '.inp' )
        if fn_trj is not None:
            _FRAG.TRAJECTORY.write( fn_trj + '_fragment_%03d' % _i, fmt = 'cpmd', sort_atoms = False )

if(__name__ == "__main__"):
    main()
