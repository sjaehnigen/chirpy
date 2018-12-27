#!/usr/bin/python3
import sys
import numpy as np
import tempfile
import copy 


def get_frame_traj_and_mom(TRAJECTORY, MOMENTS, n_atoms, n_moms): #by Arne Scherrer
    """iterates over TRAJECTORY and MOMENTS files and yields generator of positions, velocities and moments (in a.u.)
       script originally by Arne Scherrer
"""
    with open(TRAJECTORY, 'r') as traj_f, open(MOMENTS, 'r') as moms_f:
        traj_it = (list(map(float,line.strip().split()[1:])) for line in traj_f if 'NEW DATA' not in line)
        moms_it = (list(map(float,line.strip().split()[1:])) for line in moms_f if 'NEW DATA' not in line)
        try:
            while traj_it and moms_it:
                pos_au, vel_au = tuple(np.array([next(traj_it) for i_atom in range(n_atoms)]).reshape((n_atoms, 2, 3)).swapaxes(0,1))
                wc_au, c_au, m_au = tuple(np.array([next(moms_it) for i_mom in range(n_moms)]).reshape((n_moms, 3, 3)).swapaxes(0,1))
                yield pos_au, vel_au, wc_au, c_au, m_au
        except StopIteration:
            pass


def extract_mtm_data_tmp(MTM_DATA_E0,MTM_DATA_R1,n_frames,n_states): #temporary version for debugging MTM. Demands CPMD3 output file.
    fn_buf1 = tempfile.TemporaryFile(dir='/tmp/')
    fn_buf2 = tempfile.TemporaryFile(dir='/tmp/')
    
    buf1 = np.memmap(fn_buf1,dtype='float64',mode='w+',shape=(n_frames*n_states*n_states))
    with open(MTM_DATA_E0, 'r') as f:
        for i,line in enumerate(f): buf1[i]=float(line.strip().split()[-1])

    buf2 = np.memmap(fn_buf2,dtype='float64',mode='w+',shape=(n_frames*n_states*n_states,3))             
    with open(MTM_DATA_R1, 'r') as f:
        for i,line in enumerate(f): buf2[i] = np.array(line.strip().split()[-3:]).astype(float)
    
    E0 = buf1.reshape((n_frames,n_states,n_states))                    
    R1 = buf2.reshape((n_frames,n_states,n_states,3)) #mode has to be 'MD' !
    
    del buf1, buf2
    
#Factor 2 already in CPMD --> change routine later
    return E0/2,R1/2


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
}


class CPMDjob( ):

################# one class per CPMD section ##############################################################

    class _SECTION( ):
        def __init__( self, **kwargs ):
            for _i in kwargs:
                setattr(self, _i, kwargs[ _i ] )
# problem with white spaces
#            if hasattr( self, 'options' ):
#                for _o in self.options:
#                    setattr( self, _o, self.options[ _o ] )


    class _KEYWORDSECTION( _SECTION ):
        def print_section( self ):
            print( "&%s" % self.__class__.__name__ )
            for _o in self.options:
                print( " " + " ".join( [ _o ] + [ _a for _a in self.options[ _o ][ 0 ] ] ) )
                if self.options[ _o ][ 1 ] is not None: print( "  " + " ".join( [ str( _a ) for _a in self.options[ _o ][ 1 ] ] ) )
            print( "&END" )
            pass # if dict logic universal, print method universal: print key word (upper and "_" --> " ") + OPtion

        #This is a universal class that parses all possible keywords. Each derived class may contain a test routine that checks on superfluous or missing keywords
        @classmethod
        def _parse_section_input( cls, _section_input ): 

            def _parse_keyword_input( _l ):
                _keyword = _l
                _arg = []
                _nextline = None
                _section_keys = _cpmd_keyword_logic[ cls.__name__ ]
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

            return cls( options = options )


    class ATOMS( _SECTION ):
        @classmethod
        def _parse_section_input( cls, _atoms ): # this section is connected to keywords SYSTEM>ANGSTROM, ... ! ===> problem?
            #_C = {}
            kinds = []
            channels = []
            n_kinds = []
            data = []
            for _l in _atoms:
                if '*' in _l:
                    kinds.append( _l )#.split( '_' )
                    channels.append( next( _atoms ) )#.split( ) # not clean, ehat if LOC and LMAX in different oder? Remove "="
                    n = int( next( _atoms ) ) 
                    n_kinds.append( n )
                    data.append( np.array( [ list( map( float, next( _atoms ).split( ) ) ) for _i in range( n ) ] ) )
    
    #        return cls( *_C )
            return cls( kinds = kinds, channels = channels, n_kinds = n_kinds, data = data )

        def print_section( self ):
            format = '%20.10f'*3
            print( "&ATOMS" )
#        f.write(" ISOTOPE\n")
#        for elem in elems:
#            print("  %s" % elems[elem]['MASS'])
            for _k, _ch, _n, _d  in zip( self.kinds, self.channels, self.n_kinds, self.data ):
                print( "%s" % _k )
                print( " %s" % _ch )
                print( "%4d" % _n )
                for _dd in _d:
                    print( format % tuple( _dd ) )
                    #print( format % tuple( [ c for c in elems[ elem ] ['c' ][ i ] ] ) ) #cp2k format as in pythonbase
            print( "&END" )
    

        def _test_section( self ):
            pass
#    if pos_au.shape[0] != len(symbols):
#        print('ERROR: symbols and positions are not consistent!')
#        sys.exit(1)
#
#    pos = copy.deepcopy(pos_au) #copy really necessary? 
#    if fmt=='angstrom': pos /= Angstrom2Bohr
#
#    for i,sym in enumerate(symbols):
#        if sym != sorted(symbols)[i]:
#            print('ERROR: Atom list not sorted!')
#            sys.exit(1)
#        try:
#            elems[sym]['n'] +=1
#            elems[sym]['c'][elems[sym]['n']] = pos[i]
#        except KeyError:
##            if sym in constants.symbols:
##                elems[sym] = constants.species[sym]
#            elems[sym] = OrderedDict()
#            elems[sym]['n'] = 1
#            elems[sym]['c'] = {elems[sym]['n'] : pos[i]}
##            else: raise Exception("Element %s not found!" % sym)

    
    class INFO( _KEYWORDSECTION ):
        pass

    class CPMD( _KEYWORDSECTION ):
        pass

    class RESP( _KEYWORDSECTION ):
        pass

    class DFT( _KEYWORDSECTION ):
        pass

    class SYSTEM( _KEYWORDSECTION ):
        pass

########################################################################################################

    def __init__( self, **kwargs ):
        #Todo initialise with a default cpmd job (SCF)
        #mandatory section
        self.INFO = kwargs.get( 'INFO', '' )
        self.CPMD = kwargs.get( 'CPMD', {} )
        self.SYSTEM = kwargs.get( 'SYSTEM', {} )
        self.ATOMS = kwargs.get( 'ATOMS', {} )
        self.DFT = kwargs.get( 'DFT', {} )

        #optional section
        self.RESP = kwargs.get( 'RESP' )

        self._check_consistency()


    def _check_consistency( self ):
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
    
        CONTENT = { _C : getattr( CPMDjob, _C )._parse_section_input( CONTENT[ _C ] ) for _C in CONTENT }

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

#EOF
