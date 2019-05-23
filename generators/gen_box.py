#!/usr/bin/env python3.6
#Version important as <3.6 gives problems with OrderedDictionaries

import sys
import os
import copy
import numpy as np

from topology.dissection import assign_molecule
from topology.mapping import get_atom_spread

from classes.trajectory import XYZFrame
from classes.system import Supercell
from physics import constants

#put this into new lib file
#valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
#masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
#Angstrom2Bohr = 1.8897261247828971
#np.set_printoptions(precision=5,suppress=True)

class _BoxObject():

#volume is determined by _cell_vec_aa() / cell_vec_aa()
#methods that are allowed to manipulate _cell_vec_aa and/or volume_aa3: __init__, __pow__ (via __mul__)
#plan: check routine compares _method() to method and complains if it deviates
#Symmetry should be determined from cell_aa

    def __init__( self, **kwargs ): #empty-box init allowed (bare)
        self.members = kwargs.get( "members", [ ] ) # list of ( n, XYZFrame object ) tuples with n being the no. of molecules within the box
        self.symmetry = kwargs.get( 'symmetry', 'orthorhombic' )
        self.origin_aa = kwargs.get( 'origin_aa', np.zeros( ( 3 ) ).astype( float ) )
        if self.symmetry != 'orthorhombic': raise NotImplementedError( 'ERROR: Only supports orthorhombic cells' )
        self.pbc = kwargs.get( 'pbc', True )
        _BoxObject._sync_class( self ) #important: carry out here only native sync
        self.cell_vec_aa = self._cell_vec_aa( **kwargs )
        self.volume_aa3 = self._volume_aa3()
        self._clean_members()

    @classmethod
    def read( cls, fn, **kwargs ):
        #Beta Version because Supercell Class is still a little weird and chaotic (+ is_equal routine incomplete)
        #cell_aa?
        #override
        kwargs[ 'wrap_mols' ] = True
        _sys = Supercell( fn, **kwargs )

        #some corrections due to deprecated Supercell structure
        try:
            _sys.mol_map 
        except AttributeError:
            _sys.install_molecular_origin_gauge( )
        nargs = {}
        try:
            nargs[ 'cell_aa' ] = _sys.cell_aa_deg 
        except AttributeError:
            #NB: if kwargs has cell_aa attribute, _sys has to have it as well ==> no 2nd check for 'cell_aa' necessary
            print( 'WARNING: Could not find cell parametres; uses guess from atom spread!' )
            nargs[ 'cell_aa' ] = np.array( get_atom_spread( _sys.XYZData.data ) )

        #pass pbc? cell? WARNING: do not just pass kwargs! reassemble it!!
        return cls( members = [ ( 1, _s._to_frame() ) for _s in _sys.XYZData._split( _sys.mol_map ) ], **nargs )

    def _cell_vec_aa( self, **kwargs ):
        if self.symmetry == 'orthorhombic': #should be cubic
            return np.diag( kwargs.get( 'cell_aa', np.zeros( ( 3 ) ).astype( float ) )[ :3 ] )

    def _volume_aa3( self ):
        return np.dot( self.cell_vec_aa[ 0 ], np.cross( self.cell_vec_aa[ 1 ], self.cell_vec_aa[ 2 ] ) )

    def _sync_class( self ): #calculates intensive properties, does not calculates extensive properties (?)
        self.n_members = len( self.members )
        self.mass_amu = sum( [ _n * _m.masses_amu.sum() for _n, _m in self.members ] )
        self.n_atoms = sum( [ _n * _m.n_atoms for _n, _m in self.members ] )
    #def routine: check all xx attributes against _xx() methods
        
    def _clean_members( self ):
        if self.n_members == 0: return None
        _eq = np.zeros( ( self.n_members, ) * 2 )
        for _ii, ( _i, _m ) in enumerate( self.members ):
            _eq[ _ii, _ii: ] = np.array( [ bool( _m._is_equal( _n, atol = 1.0 )[ 0 ] ) for _j, _n in self.members[ _ii: ] ] )
        #_eq = np.array( [ [ bool( _m._is_equal( _n, atol = 1.0 )[ 0 ] ) for _j, _n in self.members[ _ii: ] ] for _ii, ( _i, _m ) in enumerate( self.members ) ] ).astype( bool )
        _N = self.n_members
        _M = self.n_members
        _ass = np.zeros( ( _N ) ).astype( int )
        _sp = 0
        for _im in range( _N ):
            if _ass[ _im ] == 0:
                _sp += 1
                _ass, _M = assign_molecule( _ass, _sp, _N, _eq, _im, _M )
            if _M == 0:
                break
        _n, _m = np.array( self.members ).T
        self.members = [ ( _n[ _ass == _i ].sum(), _m[ _ass == _i ][ 0 ] ) for _i in np.unique( _ass ) ]
        self._sync_class()

    def __add__( self, other ):
        if not isinstance( other, _BoxObject ): 
            raise TypeError( 'unsupported operand type(s) for +: \'%s\' and \'%s\'' % ( type( self ).__name__, type( other ).__name__ ) )
        new = copy.deepcopy( self )
        if np.allclose( self.cell_vec_aa, other.cell_vec_aa ):
            new.members += other.members
        else: raise AttributeError( 'The two objects have different cell attributes!' )
        new._sync_class()
        new._clean_members()
        return new
        #Later: choose largest cell param and lowest symmetry 

    def __mul__( self, other ):
        new = copy.deepcopy( self )
        if isinstance( other, int ): 
            for _i in range( other-1 ):
                new += self
        #elif isinstance( other, _BoxObject ): 
        # ToDo: connect it to pow    
        else:
            raise TypeError( 'unsupported operand type(s) for *: \'%s\' and \'%s\'' % ( type( self ).__name__, type( other ).__name__ ) )
        return new
        
    def __pow__( self, other ):
        print( '\nWARNING: Power in beta state. Proceed with care!\n' )
        if not isinstance( other, int ): 
            raise TypeError( 'unsupported operand type(s) for *: \'%s\' and \'%s\'' % ( type( self ).__name__, type( other ).__name__ ) )
        new = copy.deepcopy( self )
        new.members *= other
        new._cell_vec_aa = lambda : other ** (1/3) * self._cell_vec_aa()
        new.cell_vec_aa = new._cell_vec_aa()
        new.volume_aa3 = new._volume_aa3()
        new._sync_class()
        new._clean_members()
        return new

    def __radd__( self, other ): 
        return self.__add__( other )

    def __iadd__( self, other ): #using __add__ implies an (unnecessary) deepcopy of self (is python smart enough to recognise this?)
        self = self.__add__( other )
        return self

    def __rmul__( self, other ):
        return self.__mul__( other )

    def __imul__( self, other ):
        self = self.__mul__( other )
        return self

    def __ipow__( self, other ):
        self = self.__pow__( other )
        return self

    def print_info( self ):
        #Work in progress...
        print( '%12s' % self.__class__.__name__ )
        print( '%12d Members\n%12d Atoms\n%12.4f amu\n%12.4f aa3' %  ( self.n_members, self.n_atoms, self.mass_amu, self.volume_aa3 ) )
        print( 67 * '–' )
        print( ' x '.join( map( '{:.5f} aa'.format, np.dot( np.ones( ( 3 ) ), self.cell_vec_aa ) ) ) ) #simple, only for orthorhombic
        print( 67 * '–' )
        print( '%45s %8s %12s' % ( 'File', 'No.', 'Molar Mass' ) )
        print( 67 * '–' )
        print( '\n'.join( [ '%45s %8d %12.4f' % ( _m[ 1 ].fn, _m[ 0 ], _m[ 1 ].masses_amu.sum() ) for _m in self.members ] ) )
        print( 67 * '–' )

    def create_system( self, **kwargs ): #most important class (must not be adapted within derived classes)
        #work in progress... #creates a system object (Supercell)
        #needs attribute: data (or coordinates or sth ==> actual positions of the atoms)
        pass

_solvents = {}
class Solution( _BoxObject ):
#    def __new__( self, **kwargs ):
#        pass

    def __init__( self, **kwargs ):
        self.solvent = kwargs.get( "solvent" ) 
        self.rho_g_cm3 = kwargs.get ( "rho_g_cm3", 1.0 )
        self.solutes = kwargs.get( "solutes", [ ] )
        self.c_mol_L = kwargs.get( "c_mol_L", [ 1.0 ] )

        if self.solvent is None:
            print( '\nERROR: You have to specify a solvent as coordinate file or select one from the library!' )
            sys.exit( 1 )
        if self.solvent in _solvents: self.solvent = _solvents[ self.solvent ] 
        if 0. in self.c_mol_L: 
            print( '\nERROR: You have to specify non-zero values of concentration!' )
            sys.exit( 1 )

        #TMP vars
        _slt = [ XYZFrame( _s ) for _s in self.solutes ]
        _slv = XYZFrame( self.solvent )

        #--- CALCULATE INTENSIVE PROPERTIES (per 1L)
        # solvent concentration (externalise?) 
        _c_slv_mol_L = ( 1000 * self.rho_g_cm3 - sum( [ _c * _o.masses_amu.sum() for _c, _o in zip( self.c_mol_L, _slt ) ] ) ) / _slv.masses_amu.sum()
        if _c_slv_mol_L <= np.amax( self.c_mol_L ): #0
            raise ValueError( 'The given solutes\' concentrations exceed density of %.2f!\n' % self.rho_g_cm3 )
        _c_min_mol_L = np.amin( self.c_mol_L )

        #--- CALCULATE EXTENSIVE PROPERTIES
        #Get counts of solvent relative to solutes #default: set lowest c to 1
        _n = ( self.c_mol_L + [ _c_slv_mol_L ] ) / _c_min_mol_L
        def _dev_warning( _d, _id ):
            if _d > 0.01: print( '\nWARNING: Member counts differ from input value by more than 1%%:\n  - %s\n' % _id )
        [ _dev_warning( abs( round( _in ) - _in ) / _in, ( [ self.solvent ] + self.solutes )[ _ii ] ) for _ii, _in in enumerate( _n ) ] 


        _BoxObject.__init__( self, members = [ ( int( round( _in ) ), _is ) for _in, _is in zip( _n, _slt + [ _slv ] ) ] ) #by definition solvent is the last member

        del _slt, _slv, _c_slv_mol_L, _n, _c_min_mol_L, _dev_warning

    @classmethod
    def read( cls, fn, **kwargs ):
         _tmp = _BoxObject.read( fn, **kwargs )
         _out = cls.__new__( cls ) #this is unpythonic but used only temporary until __init__() routine has been adapted for read
         for _key in _tmp.__dict__:
            setattr( _out, _key, getattr( _tmp, _key ) )
         _out._sync_class()
         return _out

    def _cell_vec_aa( self, **kwargs ):
        #what to do if cell_aa argument given here?? ==> check if total volume is the same; if yes use the given cell values, otherwise raise Exception
        if self.symmetry == 'orthorhombic': #should be cubic
            return np.diag( 3 * [ ( self.mass_amu / self.rho_g_cm3 / ( constants.avog * 1E-24 ) ) ** (1/3) ] )

    def _c_mol_L( self ):
        #return [ _m[ 0 ] / ( constants.avog * 1E-27 ) / self.volume_aa3 for _m in self.members[ :-1 ] ] 
        return [ _m[ 0 ] / ( constants.avog * 1E-27 ) / self.volume_aa3 for _m in self.members ] 

    def _rho_g_cm3( self ):
        return self.mass_amu / ( constants.avog * 1E-24 ) / self.volume_aa3

    def _sync_class( self ):
        _BoxObject._sync_class( self )
        self.c_mol_L = self._c_mol_L( )
        self.rho_g_cm3 = self._rho_g_cm3( )

    def print_info( self ):
        _BoxObject.print_info( self )
        print( '%12.4f g / cm³' %  self.rho_g_cm3 )
        print( '\n'.join( map( '{:12.4f} mol / L'.format, self.c_mol_L ) ))

    def _fill_box( self ): #calls packmol
        #calculate packmol box
        _box_aa = np.concatenate( ( self.origin_aa, np.dot( np.ones( ( 3 ) ), self.cell_vec_aa ) ) )
        #if periodic: add vacuum edges
        if self.pbc: _box_aa += np.array( [ 2., 2., 2., -2., -2., -2. ] )

        with open( '.tmp_packmol.inp', 'w' ) as f:

            f.write( 'tolerance 2.000' + 2*'\n' )
            f.write( 'filetype xyz' + 2*'\n' )
            f.write( 'output .simbox.xyz' + '\n' )

            for _im, _m in enumerate( self.members ):
                _fn = '.member-%03d.xyz' % _im
                _m[ 1 ].write( _fn )
                f.write( '\n' )
                f.write( 'structure %s' % _fn + '\n' ) 
                f.write( '  number %d' % _m[ 0 ] + '\n' )
                f.write( '  inside box ' + ' '.join( map( '{:.3f}'.format, _box_aa ) )  + '\n' )
                f.write( 'end structure' + '\n' )

        #call packmol (external)
        os.system( "packmol < .tmp_packmol.inp" )
        
        #read packmol output

        #clean file
        os.remove( ".tmp_packmol.inp" )
        for _im, _m in enumerate( self.members ):
            os.remove( ".member-%03d.xyz" % _im )
        #os.remove( ".simbox.xyz" )

#class Mixture, MolecularCrystal, IonicCrystal, GasPhase( Mixture )