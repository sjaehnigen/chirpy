#!/usr/bin/env python3.6
#Version important as <3.6 gives problems with OrderedDictionaries

import sys
import os
import copy
import numpy as np

from writer.trajectory import cpmdWriter, xyzWriter, pdbWriter
from physics import constants

#put this into new lib file
valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
Angstrom2Bohr = 1.8897261247828971
np.set_printoptions(precision=5,suppress=True)

class TRAJECTORY(): #later: merge it with itertools (do not load any traj data before the actual processing)        
    def __init__( self, **kwargs ): #**kwargs for named (dict), *args for unnamed
        self.axis_pointer = kwargs.get( 'axis_pointer', 0 )

        self.comments = kwargs.get( 'comments', )
        self.symbols  = kwargs.get( 'symbols', )
        self.data     = kwargs.get( 'data', )
        self.pos_aa   = kwargs.get( 'pos_aa', )
        self.vel_au   = kwargs.get( 'vel_au', )
        self._sync_class()

    def _sync_class( self ):
        try:
            self.masses_amu = np.array( [ masses_amu[ s ] for s in self.symbols ] )
        except  KeyError:
            print('WARNING: Could not find all element masses!')
        if self.vel_au.size == 0: self.vel_au = np.zeros( self.pos_aa.shape )
        self.n_frames, self.n_atoms, self.n_fields = self.pos_aa.shape
                   
    def __add__( self, other ):
        new = copy.deepcopy( self )
        new.pos_aa = np.concatenate( ( self.pos_aa, other.pos_aa ), axis = self.axis_pointer )
        new._sync_class()
        if self.axis_pointer == 0:
            self.comments = np.concatenate((self.comments,other.comments))        
        if new.axis_pointer == 1:
            new.symbols = np.concatenate((self.symbols,other.symbols)) #not so beautiful
        return new
    
    def __iadd__( self,other ):
        self.pos_aa = np.concatenate((self.pos_aa,other.pos_aa),axis=self.axis_pointer)
        self._sync_class()
        if self.axis_pointer == 0:
            self.comments = np.concatenate((self.comments,other.comments))
        if self.axis_pointer == 1:
            self.symbols = np.concatenate((self.symbols,other.symbols))            
        return self

#    def __prod__(self,other):
#        new = copy.deepcopy(self)
#        new.data = self.data*other
#        return new
#
#    def __iprod__(self,other):
#        self.data *= other
#        return self
    
    def tail(self,n):
        new = copy.deepcopy(self)        
        new.pos_aa = self.pos_aa[-n:]
        new.n_frames = n
        return new
    
    def _sort( self ): #use set?
        elem = { s : np.where( self.symbols == s)[ 0 ] for s in np.unique( self.symbols ) }
        ind = [ i for k in sorted( elem ) for i in elem[ k ] ]
        self.pos_aa = self.pos_aa[ :, ind, : ]
        self.vel_au = self.vel_au[ :, ind, : ]
        self.symbols = self.symbols[ ind ]
        self._sync_class( )
    
    def write( self, fn, **kwargs ):
     #   if fn == self.fn:
     #       raise Exception('ERROR: write file equals source file. File would be overwritten!')
        attr = kwargs.get( 'attr', 'pos_aa' )
        factor = kwargs.get( 'factor', 1.0 ) #for velocities
        separate_files = kwargs.get( 'separate_files', False ) #only for XYZ

        if attr == 'data': self.data = np.concatenate( ( self.pos_aa, self.vel_au ), axis = -1 ) #TMP solution; NB: CPMD writes XYZ files with vel_aa 
        fmt  = kwargs.get( 'fmt', fn.split( '.' )[ -1 ] )
        if fmt == "xyz" :
            if separate_files: 
                frame_list = kwargs.get( 'frames', range( self.n_frames ) )
                [ xyzWriter( ''.join( fn.split( '. ')[ :-1 ] ) + '%03d' % fr + '.' + fn.split( '.' )[ -1 ],
                             [ getattr( self, attr )[ fr ] ], 
                             self.symbols, 
                             [ self.comments[ fr ] ] 
                           ) for fr in frame_list 
                ]
          
            else: xyzWriter( fn,
                             getattr( self, attr ),
                             self.symbols,
                             getattr( self, 'comments', self.n_frames * [ 'passed' ] ), #Writer is stupid
                           )    
        elif fmt == "pdb":
            for _attr in [ 'mol_map', 'abc', 'albega' ]: #try to conceive missing data from kwargs
                try: getattr( self,_attr )
                except AttributeError: setattr( self, _attr, kwargs.get( _attr ) )
            pdbWriter( fn,
                       self.pos_aa[ 0 ], #only frame 0 vels are not written
                       types = self.symbols,#if there are types change script
                       symbols = self.symbols,
                       residues = np.vstack( ( np.array( self.mol_map ) + 1, np.array( [ 'MOL' ] * self.symbols.shape[ 0 ] ) ) ).swapaxes( 0, 1 ), #+1 because no 0 index
                       box = np.hstack( ( self.abc, self.albega ) ),
                       title = 'Generated from %s with Molecule Class' % self.fn 
                     )

        # CPMD Writer does nor need symbols if only traj written
        elif fmt == 'cpmd': #pos and vel, attr does not apply
            print( 'CPMD WARNING: Output with sorted atomlist!' )
            loc_self = copy.deepcopy( self )
            loc_self._sort( )
            cpmdWriter( fn, loc_self.pos_aa * Angstrom2Bohr, loc_self.symbols, loc_self.vel_au * factor, **kwargs) # DEFAULTS pp='MT_BLYP', bs=''

        else: raise Exception( 'Unknown format: %s.' % fmt )

    def _is_similar( self, other ): #add more tests later
        #used by methods:
        #topology.map_atoms_by_coordinates
        f = lambda a: getattr( self, a ) == getattr( other, a )
        ie = list( map( f, [ 'n_atoms', 'n_fields' ] ) )
        ie.append( bool( np.prod( [ a == b for a, b in zip( np.sort( self.symbols ), np.sort( other.symbols ) ) ] ) ) )
        #if hasattr(data,'cell_aa')
        return np.prod( ie ), ie

