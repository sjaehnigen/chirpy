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


# LOTS of pass through variables (without checking) MOVE IT TO CLASSES when finished
class CPMDjob( ):

################# one class per CPMD section ##############################################################

    class _SECTION( ):
        def __init__( self, **kwargs ):
            for _i in kwargs:
                setattr(self, _i, kwargs[ _i ] )


    class _KEYWORDSECTION( _SECTION ):
        def print_section( self ):
            pass # if dict logic universal, print method universal: print key word (upper and "_" --> " ") + OPtion


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
            print( 'Here comes the atoms section' )
    
    
    class CPMD( _KEYWORDSECTION ):
        def __init__( self, **kwargs ):
            for _i in kwargs:
                if _i != "center_molecule": setattr(self, _i, kwargs[ _i ] )
                self.center_molecule = kwargs.get( 'center_molecule', 'OFF' )

        @classmethod
        def _parse_section_input( cls, _cpmd ): # this section is connected to keywords SYSTEM>ANGSTROM, ... ! ===> problem?
            options = []
            for _l in _cpmd: #quick and dirty
                if "CENTER MOLECULE" not in _l: options.append( _l )
                else: center_molecule = _l.split( 'CENTER MOLECULE' )[0]

            return cls( options = options, center_molecule = center_molecule )

########################################################################################################

    def __init__( self, **kwargs ):
        #mandatory section
        self.info = kwargs.get( 'info', '' )
        self.CPMD = kwargs.get( 'cpmd', {} )
        self.system = kwargs.get( 'system', {} )
        self.ATOMS = kwargs.get( 'atoms', {} )
        self.dft = kwargs.get( 'dft', {} )

        #optional section
        self.resp = kwargs.get( 'resp' )

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
            CONTENT = { _l[ 1: ].lower() : _parse_file( _iter ) for _l in _iter if "&" in _l }
    
        # parse cpmd section ( CENTER MOLECULE OFF )
        CONTENT[ 'cpmd' ] = CPMDjob.CPMD._parse_section_input( CONTENT[ 'cpmd' ] )

        # parse atoms section ( delete atoms )
        CONTENT[ 'atoms' ] = CPMDjob.ATOMS._parse_section_input( CONTENT[ 'atoms' ] )

        # for _DONE in [ 'atoms', 'cpmd' ]: del CONTENT[ _DONE ]
        # pass through remaining sections (NB: list_iterator will be CONSUMED, i.e can be read out only once ) 
        for _C in CONTENT: 
            if _C not in [ 'atoms', 'cpmd' ]: CONTENT[ _C ] = [ _c for _c in CONTENT[ _C ] ]

        return cls( **CONTENT )
    

    def write_input_file( self, fn ):
        ''' CPMD 4 '''
        #known sections and order
        _SEC = [ 'info', 'CPMD', 'resp', 'dft', 'system', 'ATOMS' ] 

        _stdout = sys.stdout # sys.__stdout__ is not Jupyter Output, so use this way to restore stdout
        sys.stdout = open( fn, 'w' )

        for _s in _SEC:
            if hasattr( self, _s ): 
                print( '&' + _s.upper() )
                try:
                    getattr( self, _s ).print_section( )
                except AttributeError:
                    pass
                print( '&END' )

        sys.stdout.flush()
        #sys.stdout = sys.__stdout__
        sys.stdout = _stdout

#EOF
