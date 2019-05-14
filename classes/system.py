#!/usr/bin/env python3.6
#Version important as <3.6 gives problems with OrderedDictionaries

#import unittest
#import logging
#import filecmp
#import types
#import filecmp
#
#from lib import debug
#from collections import OrderedDict

import sys
import os
import copy
import numpy as np

from reader.trajectory import pdbReader,xyzReader
from writer.trajectory import cpmdWriter,xyzWriter,pdbWriter
from reader.modes import xvibsReader
#from writer.modes import xvibsWriter
from interfaces import cpmd as cpmd_n #new libraries
from classes.crystal import UnitCell
from classes.trajectory import XYZData, VibrationalModes
from physics import constants
from physics.classical_electrodynamics import current_dipole_moment,magnetic_dipole_shift_origin
from physics.modern_theory_of_magnetisation import calculate_mic
from physics.statistical_mechanics import CalculateKineticEnergies #wrong taxonomy (lowercase)
from topology.dissection import define_molecules,dec
from topology.mapping import align_atoms

#put this into new lib file
valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400}
Angstrom2Bohr = 1.8897261247828971
np.set_printoptions(precision=5,suppress=True)

#switch function which should be used in future for the standard file readers instead of elif... 
#def f(x):
#    return {
#        'a': 1,
#        'b': 2
#    }.get(x, 9)    # 9 is default if x not found

#NEEDS
# - CPMD format Reader
# - Tidying up
# - USE XYZData for file reading and extract further data for Molecule from it ( ? )

class SYSTEM( ):
    def __init__(self,fn,**kwargs):
        if int(np.version.version.split('.')[1]) < 14:
            print('ERROR: You have to use a numpy version >= 1.14.0! You are using %s.'%np.version.version)
            sys.exit(1)
        global ignore_warnings
        ignore_warnings = kwargs.get('ignore_warnings',False)
        fmt = kwargs.get('fmt',fn.split('.')[-1])

        ## This is a cheap workaround
        #TOPOLOGY first, COORDINATES second: 
        if kwargs.get('fn_topo') is not None:
            print('Found topology file.')
            self.install_molecular_origin_gauge(**kwargs) #Do it as default
            #print('I wrap molecules')
            #self.XYZData._wrap_molecules(self.mol_map,self.UnitCell.abc,albega=self.UnitCell.albega)

        if fmt=="xyz":
            self.XYZData = XYZData(fn,**kwargs)
        elif fmt=="xvibs":
            mw = kwargs.get('mw',False)
            n_atoms, numbers, coords_aa, n_modes, omega_cgs, modes = xvibsReader(fn)
            symbols  = [constants.symbols[z-1] for z in numbers]
            masses   = [masses_amu[s] for s in symbols]
            if mw:
                print('Assuming mass-weighted coordinates in XVIBS.')
                modes /= np.sqrt(masses)[None,:,None]*np.sqrt(constants.m_amu_au)
            else:
                print('Not assuming mass-weighted coordinates in XVIBS (use mw=True otherwise).')
            self.XYZData = XYZData(data = coords_aa.reshape( (1, n_atoms, 3 ) ),symbols = symbols, **kwargs )
            self.Modes = VibrationalModes(fn,modes=modes,numbers=numbers,omega_cgs=omega_cgs,coords_aa=coords_aa,**kwargs)
        elif fmt=="pdb":
            data, types, symbols, residues, box_aa_deg, title = pdbReader(fn)
            n_atoms=symbols.shape[0]
            self.XYZData = XYZData(data = data.reshape( ( 1, n_atoms, 3 ) ), symbols = symbols, **kwargs )
            setattr( self, 'cell_aa_deg', kwargs.get( 'cell_aa', box_aa_deg ) )
            #Disabled 2018-12-04
            #print('Found PDB: Automatic installation of molecular gauge.')
            #self.install_molecular_origin_gauge(fn_topo=fn) #re-reads pdb file

        else: raise Exception('Unknown format: %s.'%fmt)

        cell_aa_deg = kwargs.get('cell_aa',getattr(self,"cell_aa_deg",None)) 
        if cell_aa_deg is not None:
            if hasattr(self,'cell_aa_deg'):
                if not np.allclose(cell_aa_deg,self.cell_aa_deg): print('WARNING: Given cell size differs from file parametres!')
            self.UnitCell = UnitCell(cell_aa_deg)
            if kwargs.get('cell_multiply') is not None:
                cell_multiply = kwargs.get('cell_multiply')
                cell_priority = kwargs.get('cell_priority',(2,0,1)) #priority from CPMD (monoclinic)
                self.XYZDataUnitCell = copy.deepcopy(self.XYZData)
                self.XYZData = self.UnitCell.propagate(self.XYZData,cell_multiply,priority=cell_priority) #priority from CPMD (monoclinic)
                if hasattr(self,'Modes'): 
                    self.ModesUnitCell = copy.deepcopy(self.Modes)
                    self.Modes = self.UnitCell.propagate(self.Modes,cell_multiply,priority=cell_priority) #priority from CPMD (monoclinic)
            if kwargs.get('wrap_mols') is not None:
                print('I wrap molecules')
                #ADDED 2018-11-28
                if not hasattr(self,'mol_map'): self.install_molecular_origin_gauge()
                self.XYZData._wrap_molecules(self.mol_map,cell_aa_deg,**kwargs)

            #DEPENDING on cell_aa?
            center_res = kwargs.get('center_residue',-1) #-1 means False, has to be integer ##==> ToDo if time: elaborate this method (maybe class independnet as symmetry tool)
            if center_res != -1: #is not None
                print('Centering residue %d in cell. Auto-wrapping of residues.'%center_res)
                if not hasattr(self,'mol_map'): raise Exception('ERROR: System does not recognise any residues/molecules!')
                self.XYZData._wrap_molecules(self.mol_map,cell_aa_deg,**kwargs)
                self.XYZData._move_residue_to_centre(center_res,cell_aa_deg,**kwargs)
                self.XYZData._wrap_molecules(self.mol_map,cell_aa_deg,**kwargs)

    def install_molecular_origin_gauge(self,**kwargs):
        '''Script mainly from Arne Scherrer'''
        fn = kwargs.get('fn_topo')
        if fn: #use pdbReader
            data, types, symbols, residues, box_aa_deg, title = pdbReader(fn)
            #convert to n_map
            resi = ['-'.join(_r) for _r in residues]
            _map_dict = dict(zip(list(dict.fromkeys(resi)),range(len(set(resi))))) #python>=3.6: keeps order
            n_map = [_map_dict[_r] for _r in resi]
            #n_map = residues[:,0].astype(int).tolist() #-1 as workaround


            setattr(self,'cell_aa_deg',kwargs.get('cell_aa',box_aa_deg))
#            n_map, symbols, cell_au = list(), list(), None
#            for line in [l.split() for l in open(fn, 'r')]:
#                if 'ATOM' in line:
#                    n_map.append(int(line[4])-1), symbols.append(line[-1]) 
#                elif 'CRYST1' in line:
#                    if not 'cell_aa' in kwargs:
#                        kwargs['cell_aa'] = np.array([Angstrom2Bohr*e for e in map(float,line[1:4])])
            cell_au = np.array([Angstrom2Bohr*e for e in box_aa_deg[:3]])
            if cell_au.any() == None:
                raise Exception('Cell has to be specified, only orthorhombic cells supported!')
            if hasattr(self,'UnitCell'):
                abc = self.UnitCell.abc
                if not np.allclose(cell_au,abc*Angstrom2Bohr):
                    raise Exception('Unit cell parametres of Molecule do not agree with topology file!')
            #n_atoms, n_mols, n_moms = len(symbols), max(n_map)+1, sum(ZV)//2
            n_atoms, n_mols = symbols.shape[0], max(n_map)+1
            n_map,symbols = zip(*[(im,symbols[ia]) for ia,a in enumerate(n_map) for im,m in enumerate(kwargs.get('extract_mols',range(n_mols))) if a==m])
            #if np.array(symbols) != self.XYZData.symbols:
            #    raise Exception('Symbols of Molecule do not agree with topology file!')
        else:
            n_map = tuple(define_molecules(self)-1)

        n_mols = max(n_map)+1 
        self.mol_map = n_map
        self.n_mols  = n_mols

        #WE need a rotuine that updates XYZData independent of _wrap_mols, because we may not want the latter

       #####THIS IS only TMP as long as XYZData has no method for this! #################
       #                                                                                #
       #                                                                                #
        setattr(self.XYZData,'mol_map',n_map)
       #                                                                                #
       #                                                                                #
       #                                                                                #
       ##################################################################################


       ### if hasattr(self,'UnitCell'):dd
       ###     if hasattr(self.UnitCell,'multiply'): #This function appears frequently --> change general structure of UnitCell object
       ###         abc = self.UnitCell.abc*self.UnitCell.multiply
       ###     else:
       ###         abc = self.UnitCell.abc
        ############WRITING INTO OBJECTS FROM OUTER SCOPE IS NOT SO CLEAN############################
        ############BETTER create respective methods in objects #######################
        if hasattr(self,'Modes'):
            ZV, M = zip(*[(valence_charges[s], masses_amu[s]) for s in self.XYZData.symbols])
            ZV, M = dec(ZV, n_map), dec(M, n_map)
            #Modes
            pos_au = dec(self.Modes.pos_au, n_map)
            #wrap molecules (in order to get "correct" com), reference: first atom 0 of mol
            if hasattr(self,'UnitCell'):
                abc = self.UnitCell.abc
                pos_au = [p-np.around((p-p[0,None,:])/(abc*Angstrom2Bohr))*abc*Angstrom2Bohr for p in pos_au]
                self.Modes.cell_au = abc*Angstrom2Bohr #not so nice transfer of attributes
            # calculate molecular centers of mass
            self.Modes.mol_com_au = np.array([np.sum(pos_au[i_mol]*M[i_mol][:,None], axis=0)/M[i_mol].sum() for i_mol in range(n_mols)])
            self.Modes.mol_com_au = np.remainder(self.Modes.mol_com_au,self.Modes.cell_au)
            self.Modes.mol_map = n_map
            self.Modes.n_mols  = n_mols


#class Molecule():

class Supercell( SYSTEM ):
    pass

class Molecule( SYSTEM ):
    pass
