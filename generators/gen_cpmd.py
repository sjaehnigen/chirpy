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

import sys
import numpy as np
import copy

from ..classes.trajectory import TRAJECTORY as _TRAJ
# from ..writers.trajectory import cpmdWriter
from ..readers.trajectory import cpmdReader
from ..physics import constants

_cpmd_keyword_logic = {
    'INFO' : {
    },
    'CPMD' : {
        'CENTER MOLECULE' : ( [ '', 'OFF' ], None ) ,
        'CONVERGENCE ORBITALS' : ( [ '' ], str ) ,  #should be float, but python does not undestand number format ( e.g., 2.0D-7); missing options: number of 2nd-line arguments
        'HAMILTONIAN CUTOFF' : ( [ '' ], float ) ,
        'WANNIER PARAMETER' : ( [ '' ], str ) , # 4 next-line arguments
    },
    'RESP' : {
        'CONVERGENCE' : ( [ '' ], str ) ,
        'CG-ANALYTIC' : ( [ '' ], int ) ,
    },
    'DFT' : {
        'FUNCTIONAL' : ( [ 'BLYP', 'PBE' ], None ) ,
    },
    'SYSTEM' : {
        'SYMMETRY' : ( [ '' ], str ) ,
        'CELL' : ( ['', 'ABSOLUTE' ], float ) ,
        'CUTOFF' : ( [ '' ], float ) ,
    },
    'ATOMS' : {
    },
}


################# one class per CPMD section ##############################################################
class SECTION( ):
    def __init__( self, name, **kwargs ):
        if name not in _cpmd_keyword_logic.keys(): raise AttributeError( 'Unknown section: %s' % name )
        self.__name__ = name
        for _i in kwargs:
            setattr(self, _i, kwargs[ _i ] )

    def print_section( self ):
        print( "&%s" % self.__name__ )
        if self.__name__ == "ATOMS":
            format = '%20.10f'*3
        #f.write(" ISOTOPE\n")
        #for elem in elems:
        #    print("  %s" % elems[elem]['MASS'])
            for _k, _ch, _n, _d  in zip( self.kinds, self.channels, self.n_kinds, self.data ):
                print( "%s" % _k )
                print( " %s" % _ch )
                print( "%4d" % _n )
                for _dd in _d:
                    print( format % tuple( _dd ) )
                    #print( format % tuple( [ c for c in elems[ elem ] ['c' ][ i ] ] ) ) #cp2k format as in pythonbase
        else:
            for _o in self.options:
                print( " " + " ".join( [ _o ] + [ _a for _a in self.options[ _o ][ 0 ] ] ) )
                if self.options[ _o ][ 1 ] is not None: print( "  " + " ".join( [ str( _a ) for _a in self.options[ _o ][ 1 ] ] ) )
        print( "&END" )

    #This is a universal class that parses all possible keywords. Each derived class may contain a test routine that checks on superfluous or missing keywords
    def _parse_section_input( self, _section_input ): 
        _C = {}
        if self.__name__ == "ATOMS" :
            kinds = []
            channels = []
            n_kinds = []
            data = []
            for _l in _section_input:
                if '*' in _l:
                    kinds.append( _l )#.split( '_' )
                    channels.append( next( _section_input ) )#.split( ) # not clean, ehat if LOC and LMAX in different oder? Remove "="
                    n = int( next( _section_input ) ) 
                    n_kinds.append( n )
                    data.append( np.array( [ list( map( float, next( _section_input ).split( ) ) ) for _i in range( n ) ] ) )
            _C[ 'kinds' ] = kinds
            _C[ 'channels' ] = channels
            _C[ 'n_kinds' ] = n_kinds
            _C[ 'data' ] = data

        else:
            def _parse_keyword_input( _l ):
                _keyword = _l
                _arg = []
                _nextline = None
                _section_keys = _cpmd_keyword_logic[ self.__name__ ]
                for _k in _section_keys.keys():
                    if _k in _l:
                        _arg = _l.split( _k )[ 1 ].strip().split()
                        _arglist, _nextline = _section_keys[ _k ]
                        if any( _a not in _arglist for _a in _arg ):
                            raise Exception( 'Unknown argument for keyword %s: %s !' % ( _k, _arg ) )
                        _keyword = _k
                        break
                return _keyword, _arg, _nextline 

                    
            options = { }
            for _l in _section_input: 
                _key = _parse_keyword_input( _l )
                _nl = _key[ 2 ]
                if _key[ 2 ] is not None: _nl = list( map( _key[ 2 ], next( _section_input ).split( ) ) )
                options.update( { _key[ 0 ] : ( _key[ 1 ], _nl ) } )
            _C[ 'options' ] = options

        self.__dict__.update( _C )

        return self #a little bit strange to have to return self
        #return cls( name, **_C )

# set section defaults and read section_input

#####i USING DECORATOR
#    def get( f ):
#        @classmethod
#        def get_section( cls, **kwargs ):
#            return cls( f.__name__, **kwargs )
#        return get_section
#    @get
#    def INFO( **kwargs ):
#        pass

    def _get( name ):
        @classmethod
        def get_section( cls, **kwargs ):
            return cls( name, **kwargs )
        return get_section

    # can this be further simplified based on dict?
    # Allow empty classes ? ==> set default ==> include class test into __init__
    INFO = _get( 'INFO' )
    CPMD = _get( 'CPMD' )
    RESP = _get( 'RESP' )
    DFT = _get( 'DFT' )
    SYSTEM = _get( 'SYSTEM' )
    ATOMS = _get( 'ATOMS' )

#    @classmethod
#    def INFO( cls ):
#        return cls( 'INFO', options = { 'CPMD DEFAULT JOB' : ( [], None ) } )

#    def _test_section( self ):
#        pass
##    if pos_au.shape[0] != len(symbols):
##        print('ERROR: symbols and positions are not consistent!')
##        sys.exit(1)
##
##    pos = copy.deepcopy(pos_au) #copy really necessary? 
##    if fmt=='angstrom': pos /= Angstrom2Bohr
##
##    for i,sym in enumerate(symbols):
##        if sym != sorted(symbols)[i]:
##            print('ERROR: Atom list not sorted!')
##            sys.exit(1)
##        try:
##            elems[sym]['n'] +=1
##            elems[sym]['c'][elems[sym]['n']] = pos[i]
##        except KeyError:
###            if sym in constants.symbols:
###                elems[sym] = constants.species[sym]
##            elems[sym] = OrderedDict()
##            elems[sym]['n'] = 1
##            elems[sym]['c'] = {elems[sym]['n'] : pos[i]}
###            else: raise Exception("Element %s not found!" % sym)
########################################################################################################
class CPMD_TRAJECTORY( _TRAJ ): ##name?
    # link it to XYZData object ?
    # IDEA: write general TRAJ class and definde derived classes, solve pos_aa/ pos_au dualism (maybe by keyword?)
    '''Convention: pos in a.a., vel in a.u. // Use keyword to switch between representations '''
    # NB: CPMD writes either pos_aa + vel_aa or *_au, regardless the file format ( xyz convention here is pos_aa/vel_au though... )

    def __init__( self, **kwargs ):
        for _i in kwargs:
            setattr(self, _i, kwargs[ _i ] )
        self._sync_class( ) 
            
    @classmethod
    def read( cls, fn, n_atoms, **kwargs ): # NOT SO NICE: Get rid of n_atoms and explicit symbols (should be automatic )
        #data = tuple( [ ( _p, _v ) for _p, _v in cpmdReader( fn, n_atoms, **kwargs ) ] ) #take advantage of generator at a later instance
        data = np.array( [ _d for _d in cpmdReader( fn, n_atoms, kwargs.get( 'type', 'TRAJECTORY' ) ) ] ) #take advantage of generator at a later instance
        pos, vel = tuple( data.swapaxes( 0, 1 ) )
        return cls( pos_aa = pos * constants.l_au2aa , vel_au = vel, symbols = kwargs.get( 'symbols' ) )  

#    def write( self, fn, **kwargs ): #Later: Use global write function of _TRAJ
#        sym = [ 'X' ] * self.pos.shape[ 1 ]
#        # ToDo: updat..writers: symbols not needed for TRAJSAVED output
#        cpmdWriter( fn, self.pos, sym, self.vel, write_atoms = False )

class CPMDjob( ):
    def __init__( self, **kwargs ):
        #Todo initialise with a default cpmd job (SCF) ==> where to put defaults?
        #mandatory section
        #ToDo just pass **kwargs to respective section init
        self.INFO = kwargs.get( 'INFO', SECTION.INFO( options = { 'CPMD DEFAULT JOB' : ( [], None ) } ) ) #( 'INFO', options = { 'CPMD DEFAULT JOB' : ( [], None ) } ) )
        self.CPMD = kwargs.get( 'CPMD', {} )
        self.SYSTEM = kwargs.get( 'SYSTEM', {} )
        self.ATOMS = kwargs.get( 'ATOMS', {} )
        self.DFT = kwargs.get( 'DFT', {} )

        #optional section
        self.RESP = kwargs.get( 'RESP' )
        self.TRAJECTORY = kwargs.get( 'TRAJECTORY' )

        self._check_consistency()


    def _check_consistency( self ):
#        if any( [ not hasattr( self, _s ) for _s in [ ... ] ] ): raise TypeError: ...
        #check if all SECTIONS are actual SECTIONS
        pass

    @classmethod
    def read_input_file( cls, fn ):
        ''' CPMD 4 '''

        def _parse_file( _iter ):
            _l = next( _iter )
            _c = []
            while "&END" not in _l:
                _c.append( _l )
                _l = next( _iter )
            return iter( _c )
    
        # Parse file
        with open( fn, 'r' ) as _f:
            _iter = ( _l.strip( ) for _l in _f )
            CONTENT = { _l[ 1: ].upper() : _parse_file( _iter ) for _l in _iter if "&" in _l }
    
        #CONTENT = { _C : getattr( CPMDjob, _C )._parse_section_input( CONTENT[ _C ] ) for _C in CONTENT }
        #CONTENT = { _C : SECTION._parse_section_input( _C, CONTENT[ _C ] ) for _C in CONTENT }
        CONTENT = { _C : getattr( SECTION, _C )()._parse_section_input( CONTENT[ _C ] ) for _C in CONTENT }

        return cls( **CONTENT )
    

    def write_input_file( self, *args ):
        ''' CPMD 4 '''

        _stdout = sys.stdout # sys.__stdout__ is not Jupyter Output, so use this way to restore stdout
        if len(args) == 1:
            sys.stdout = open( args[ 0 ], 'w' )
        elif len(args) > 1: raise TypeError( self.write_input_file.__name__ + ' takes at most 1 argument.' )

        #known sections and order
        _SEC = [ 'INFO', 'CPMD', 'RESP', 'DFT', 'SYSTEM', 'ATOMS' ] 

        for _s in _SEC:
            if hasattr( self, _s ): 
                getattr( self, _s ).print_section( )

        sys.stdout.flush()
        #sys.stdout = sys.__stdout__
        sys.stdout = _stdout

    def get_positions( self ):
        ''' in a. u. ? '''
        return np.vstack( self.ATOMS.data )

    def get_symbols( self ):
        symbols = []
        for _s, _n in zip( self.ATOMS.kinds, self.ATOMS.n_kinds ): symbols +=  _s.split( '_' )[ 0 ][ 1: ]  * _n        
        return np.array( symbols )

    def get_kinds( self ):
        kinds = []
        for _s, _n in zip( self.ATOMS.kinds, self.ATOMS.n_kinds ): kinds += [ _s ] * _n        
        return np.array( kinds )

    def get_channels( self ):
        channels = []
        for _s, _n in zip( self.ATOMS.channels, self.ATOMS.n_kinds ): channels += [ _s ] * _n        
        return np.array( channels )

    def split_atoms( self, ids ):
        ''' Split atomic system into fragments and create CPMD job, respectively.
            ids ... list  of fragment id per atom (can be anything str, int, float, ... ) '''
        _dec = lambda _P: [ [ _P[ _k ] for _k, _jd in enumerate( ids ) if _jd == _id ] for _id in sorted( set( ids ) ) ]
        # problem: set is UNSORTED ==> see TRAJECTORY._sort routine for another way

        _L = []
        
        if hasattr( self, 'TRAJECTORY' ): _getlist = [ 
                                                      self.get_positions( ),
                                                      self.get_kinds( ), 
                                                      self.get_channels( ),
                                                      self.TRAJECTORY.pos_aa.swapaxes( 0, 1 ),
                                                      self.TRAJECTORY.vel_au.swapaxes( 0, 1 ),
                                                     ]
        else: _getlist =[ 
                         self.get_positions( ),
                         self.get_kinds( ), 
                         self.get_channels( ),
                        ]

#        for _sd, _sk, _sch in zip( *map( _dec, _getlist ) ):
        for _frag in zip( *map( _dec, _getlist ) ):
            _C = {}
            _C[ 'kinds' ] = []
            _C[ 'channels' ] = []
            _C[ 'n_kinds' ] = []
            _C[ 'data' ] = []
            #complicated to keep order
            _isk_old = None
            #for _I, _isk in enumerate( sorted( set( _frag[ 1 ] ) ) ):
            for _isk in  _frag[ 1 ]:
                if _isk != _isk_old:
                    _C[ 'kinds' ].append( _isk )
                    _C[ 'channels' ].append( [ _frag[ 2 ][ _j ] for _j, _jsk in enumerate( _frag[ 1 ] ) if _jsk == _isk ] [ 0 ] )# a little awkward
                    _C[ 'n_kinds' ].append( _frag[ 1 ].count( _isk ) )
                    _C[ 'data' ].append( np.array( [ _frag[ 0 ][ _j ] for _j, _jsk in enumerate( _frag[ 1 ] ) if _jsk == _isk ] ) )
                _isk_old = copy.deepcopy( _isk ) #necessary?

            out = copy.deepcopy( self )
            out.ATOMS = SECTION( 'ATOMS' , **_C )

            if hasattr( self, 'TRAJECTORY' ):
                out.TRAJECTORY = CPMD_TRAJECTORY( pos_aa = np.array( _frag[ 3 ] ).swapaxes( 0, 1 ), 
                                                  vel_au = np.array( _frag[ 4 ] ).swapaxes( 0, 1 ),
                                                  symbols = out.get_symbols( ),
                                                )

            _L.append( out )

            #print( out.TRAJECTORY.pos )
            #print( out.TRAJECTORY.pos.shape )
            #out.ATOMS.print_section()

        return _L

#EOF
