#!/usr/bin/env python3.6
#Version important as <3.6 gives problems with OrderedDictionaries

import sys
import os
import copy
import numpy as np

from topology.dissection import assign_molecule

from classes.trajectory import XYZFrame

#from physics import constants

#put this into new lib file
#valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
#masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
#Angstrom2Bohr = 1.8897261247828971
#np.set_printoptions(precision=5,suppress=True)

class _BoxObject():
    def __init__( self, **kwargs ): #empty-box init allowed (bare)
        self.members = kwargs.get( "members", [ ] ) # list of ( n, XYZFrame object ) tuples with n being the no. of molecules within the box
        self.cell_vec_aa = kwargs.get( 'cell_vec_aa', np.zeros( ( 3, 3 ) ).astype( float ) )
        self._volume_aa3 = np.dot( self.cell_vec_aa[ 0 ], np.cross( self.cell_vec_aa[ 1 ], self.cell_vec_aa[ 2 ] ) )
        self.symmetry = kwargs.get( 'symmetry', 'orthorhombic' )
        if self.symmetry != 'orthorhombic': raise NotImplementedError( 'ERROR: Only supports orthorhombic cells' )
        self._sync_class( **kwargs )
        self._clean_members()

    #def routine: check, update and tell

    def _sync_class( self, **kwargs ): #calculates intensive properties, does not calculates extensive properties (?)
        self.n_members = len( self.members )
        self.mass_amu = sum( [ _n * _m.masses_amu.sum() for _n, _m in self.members ] )
        self.n_atoms = sum( [ _n * _m.n_atoms for _n, _m in self.members ] )

        #apply hidden values, any changes?
        self.volume_aa3 = self._volume_aa3
        #self.voxel     = np.dot(self.cell_au[0],np.cross(self.cell_au[1],self.cell_au[2]))
        
    def _clean_members( self ):
        if self.n_members == 0: return None
        _eq = np.array( [ [ bool( _m._is_equal( _n )[ 0 ] ) for _j, _n in self.members ] for _i, _m in self.members ] ).astype( bool )
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
        new = copy.deepcopy( self )
        if np.allclose( self.cell_vec_aa, other.cell_vec_aa ):
            new.members += other.members
        else: raise AttributeError( 'ERROR: The two objects have different cell attributes!' )
        new._sync_class()
        new._clean_members()
        return new
        #Later: choose largest cell param and lowest symmetry 
        
    def __iadd__( self, other ):
        if np.allclose( self.cell_vec_aa, other.cell_vec_aa ):
            self.members += other.members
        else: raise AttributeError( 'ERROR: The two objects have different cell attributes!' )
        self._sync_class()
        self._clean_members()
        return self
        #Later: choose largest cell param and lowest symmetry 

    def print_info( self ):
        #Work in progress...
        print( '%12s' % self.__class__.__name__ )
        print( '%12d Members\n%12d Atoms\n%12.2f amu\n%12.2f aa3' %  ( self.n_members, self.n_atoms, self.mass_amu, self.volume_aa3 ) )
        print( 67 * '–' )
        print( '%45s %8s %12s' % ( 'File', 'No.', 'Molar Mass' ) )
        print( 67 * '–' )
        print( '\n'.join( [ '%45s %8d %12.2f' % ( _m[ 1 ].fn, _m[ 0 ], _m[ 1 ].masses_amu.sum() ) for _m in self.members ] ) )
        print( 67 * '–' )

    def create_system( self, **kwargs ): #most important class (must not be adapted within derived classes)
        #work in progress... #creates a system object (Supercell)
        pass

# class Mixture, MolecularCrystal, GasPhase, IonicCrystal

class Solution( _BoxObject ):
    def __init__( self, **kwargs ):
        self.solvent = kwargs.get( "solvent" ) 
        self.solute = kwargs.get( "solute" )
        self.c_mol_L = kwargs.get( "c_mol_L", 1.0 )
        self.rho_g_cm3 = kwargs.get ( "rho_g_cm3", 1.0 )

        _BoxObject.__init__( self, members = [] )
        #redefine _volume
        #redefine: _sync?
        # the solution object will set the solute with lowest conetration to 1 (or to get smallest int or all?)


    def _fill_box( self ): #calls packmol
        pass 


#class Mixture, MolecularCrystal, IonicCrystal, GasPhase( Mixture )
