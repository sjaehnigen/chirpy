#!/usr/bin/env python3.6
#Version important as <3.6 gives problems with OrderedDictionaries

import sys
import os
import copy
import numpy as np

#from reader.trajectory import pdbReader,xyzReader
#from writer.trajectory import cpmdWriter,xyzWriter,pdbWriter
#from reader.modes import xvibsReader
##from writer.modes import xvibsWriter
#from interfaces import cpmd as cpmd_n #new libraries
#from classes.crystal import UnitCell
#from classes.trajectory import XYZData
#from physics import constants
#from physics.classical_electrodynamics import current_dipole_moment,magnetic_dipole_shift_origin
#from physics.modern_theory_of_magnetisation import calculate_mic
#from physics.statistical_mechanics import CalculateKineticEnergies #wrong taxonomy (lowercase)
#from topology.dissection import define_molecules,dec
#from topology.mapping import align_atoms

#put this into new lib file
#valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
#masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
#Angstrom2Bohr = 1.8897261247828971
#np.set_printoptions(precision=5,suppress=True)

class BoxObject():
    def __init__( self, **kwargs ):
        #self.symmetry
        self.n_atoms
        self.n_members
        self.cell_aa
        self.volume_aa3
        self.mass_amu
        self.members
        self.counts
        self.masses_amu

        self._n_atoms
        self._n_members
        self._cell_aa
        self._volume_aa3
        self._members
        self._counts
        self._masses_amu

    def _sync_class( self ): #universal?
        # apply our hidden methods? ==> universal

        # NEVER overwrite
        # density, members/solvent/solute, concentration, counts

        # the solution object will set the solute with lowest conetration to 1 (or to get smallest int or all?)

        pass
#        try:
#            self.masses_amu = np.array( [ masses_amu[ s ] for s in self.symbols ] )
#        except  KeyError:
#            print('WARNING: Could not find all element masses!')
#        if self.vel_au.size == 0: self.vel_au = np.zeros( self.pos_aa.shape )
#        self.n_frames, self.n_atoms, self.n_fields = self.pos_aa.shape


#    def __add__( self, other ):
#        new = copy.deepcopy( self )
#        new.pos_aa = np.concatenate( ( self.pos_aa, other.pos_aa ), axis = self.axis_pointer )
#        new._sync_class()
#        if self.axis_pointer == 0:
#            self.comments = np.concatenate((self.comments,other.comments))
#        if new.axis_pointer == 1:
#            new.symbols = np.concatenate((self.symbols,other.symbols)) #not so beautiful
#        return new
#
#    def __iadd__( self,other ):
#        self.pos_aa = np.concatenate((self.pos_aa,other.pos_aa),axis=self.axis_pointer)
#        self._sync_class()
#        if self.axis_pointer == 0:
#            self.comments = np.concatenate((self.comments,other.comments))
#        if self.axis_pointer == 1:
#            self.symbols = np.concatenate((self.symbols,other.symbols))
#        return self


    def print_info():
        #print class name
        pass

    def create_system( self, **kwargs ): #most important class (must not be adapted within derived classes)
        pass

# class Mixture, MolecularCrystal, GasPhase, IonicCrystal

class Solution( BoxObject ):
    def __init__( self, **kwargs ):
        self.solvent
        self.solute
        self.concentration_mol_L
        self.density_g_cm3


    def _fill_box( self ): #calls packmol
        pass 

#   def _sync_class( self ):
#        try:
#            self.masses_amu = np.array( [ masses_amu[ s ] for s in self.symbols ] )
#        except  KeyError:
#            print('WARNING: Could not find all element masses!')
#        if self.vel_au.size == 0: self.vel_au = np.zeros( self.pos_aa.shape )
#        self.n_frames, self.n_atoms, self.n_fields = self.pos_aa.shape
