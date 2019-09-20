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
#!/usr/bin/env python
#Version important as <3.6 gives problems with OrderedDictionaries

#import unittest
#import logging
#import filecmp
#import types
#import filecmp

import sys
import copy
import numpy as np

from ..readers.trajectory import pdbReader
from ..readers.modes import xvibsReader
from ..classes.crystal import UnitCell
from ..classes.trajectory import XYZFrame, XYZTrajectory, VibrationalModes
from ..physics import constants
from ..physics.constants import masses_amu
from ..topology.dissection import define_molecules
from ..topology.mapping import dec

#put this into new lib file
valence_charges = {'H':1,'D':1,'C':4,'N':5,'O':6,'S':6}
#masses_amu = {'H': 1.00797,'D': 2.01410,'C':12.01115,'N':14.00670,'O':15.99940,'S':32.06400, 'Cl':35.45300 }
Angstrom2Bohr = 1.8897261247828971
np.set_printoptions(precision=5,suppress=True)

#switch function which should be used in future for the standard fil..readerss instead of elif... 
#def f(x):
#    return {
#        'a': 1,
#        'b': 2
#    }.get(x, 9)    # 9 is default if x not found

#NEEDS
# - CPMD format Reader
# - Tidying up
# - USE XYZData for file reading and extract further data for Molecule from it ( ? )

# ToDo: A lot!
# ToDo: PDB input does not save file as potential fn_topo (problem since disabling automatic mol gauge installation )

#ToDo: Write should also be called here (e.g. problem of mol_map )

class _SYSTEM( ):
    def __init__(self,fn,**kwargs):
        if int(np.version.version.split('.')[1]) < 14:
            print('ERROR: You have to use a numpy version >= 1.14.0! You are using %s.'%np.version.version)
            sys.exit(1)
        global ignore_warnings
        ignore_warnings = kwargs.get('ignore_warnings',False)
        fmt = kwargs.get('fmt',fn.split('.')[-1])

        #Beta: store FileReader data in dict, try if sth is there before startin..readers. Problem: memory?
        global _fn
        _fn = {}
        ## This is a cheap workaround
        #TOPOLOGY first, COORDINATES second: 

        self.mol_map = kwargs.get( "mol_map" )

        #ToDO: shift check for mol_map into self.install_molecular_origin_gauge method

        if kwargs.get('fn_topo') is not None:
            if self.mol_map is None:
                print('Found topology file.')
                self.install_molecular_origin_gauge( **kwargs ) #Do it as default
            else:
                print('Found topology file, but will not use it (mol_map given).')

        if fmt=="xyz":
            self.XYZData = self._XYZ( fn, **kwargs )
        elif fmt=="xvibs":
            mw = kwargs.get('mw',False)
            _fn[ fn ] = xvibsReader( fn )
            n_atoms, numbers, coords_aa, n_modes, omega_cgs, modes = _fn[ fn ]
            symbols  = [constants.symbols[z-1] for z in numbers]
            masses   = [masses_amu[s] for s in symbols]
            if mw:
                print('Assuming mass-weighted coordinates in XVIBS.')
                modes /= np.sqrt(masses)[None,:,None]*np.sqrt(constants.m_amu_au)
            else:
                print('Not assuming mass-weighted coordinates in XVIBS (use mw=True otherwise).')
            self.XYZData = self._XYZ( data = coords_aa.reshape( (1, n_atoms, 3 ) ), symbols = symbols, **kwargs )
            self.Modes = VibrationalModes(fn,modes=modes,numbers=numbers,omega_cgs=omega_cgs,coords_aa=coords_aa,**kwargs)
        elif fmt=="pdb":
            _fn[ fn ] = pdbReader( fn )
            data, types, symbols, residues, box_aa_deg, title = _fn[ fn ]
            n_atoms = symbols.shape[ 0 ]
            self.XYZData = self._XYZ( data = data.reshape( ( 1, n_atoms, 3 ) ), symbols = symbols, **kwargs )
            setattr( self, 'cell_aa_deg', kwargs.get( 'cell_aa', box_aa_deg ) )
            #Disabled 2018-12-04/Enabled 2019-05-23 w/ condition
            #print('Found PDB: Automatic installation of molecular gauge.')
            if self.mol_map is None:
                self.install_molecular_origin_gauge( fn_topo = fn ) #re-reads pdb file

        else: raise Exception('Unknown format: %s.'%fmt)


        cell_aa_deg = kwargs.get( 'cell_aa' ) #DEPRECATED
        cell_aa_deg = kwargs.get( 'cell_aa_deg' ) #, getattr( self, "cell_aa_deg", None ) )  )
        # ToDo: Fix this unlogical
        if cell_aa_deg is not None:
            cell_aa_deg = np.array( cell_aa_deg )
            if hasattr(self,'cell_aa_deg'):
                if not np.allclose( cell_aa_deg, self.cell_aa_deg ): 
                    print( 'WARNING: Given cell size differs from file parametres!' )
            self.cell_aa_deg = cell_aa_deg

        if hasattr(self, 'cell_aa_deg'):
            try:
                self.UnitCell = UnitCell( self.cell_aa_deg )
                if kwargs.get('cell_multiply') is not None:
                    cell_multiply = kwargs.get('cell_multiply')
                    cell_priority = kwargs.get('cell_priority',(2,0,1)) #priority from CPMD (monoclinic)
                    self.XYZDataUnitCell = copy.deepcopy(self.XYZData)
                    self.XYZData = self.UnitCell.propagate(self.XYZData,cell_multiply,priority=cell_priority) #priority from CPMD (monoclinic)

                    #---------------------------------------
                    #TMP cell and mol_map is not replicated
                    self.mol_map *= int(np.prod(cell_multiply))
                    self.cell_aa_deg[ :3 ] *= np.array(cell_multiply)
                    #TMP reorder: fix it nicely
                    #_cp = list(cell_priority)
                    #self.cell_aa_deg[:3] = self.cell_aa_deg[:3][_cp] 
                    #self.cell_aa_deg[3:] = self.cell_aa_deg[3:][_cp] 
                    #---------------------------------------

                    if hasattr(self,'Modes'):
                        self.ModesUnitCell = copy.deepcopy(self.Modes)
                        self.Modes = self.UnitCell.propagate(self.Modes,cell_multiply,priority=cell_priority) #priority from CPMD (monoclinic)
            except TypeError: #is this the correct Exception?
                pass

            if kwargs.get( 'wrap_mols', False ):
                if self.mol_map is None: 
                    self.install_molecular_origin_gauge()
                self.wrap_molecules() 

            center_res = kwargs.get('center_residue',-1) #-1 means False, has to be integer ##==> ToDo if time: elaborate this method (maybe class independnet as symmetry tool)
            if center_res != -1: #is not None
                self.wrap_molecules() 
                self.XYZData._center_position(self.mol_c_aa[center_res], self.cell_aa_deg)#, **kwargs )
                self.wrap_molecules() 

        # ToDo: awkward workaround (needed if XYZData._wrap_molecules() has never been called)
        if self.mol_map is not None:
            self.XYZData.mol_map = self.mol_map

    def wrap_molecules(self):
        self.mol_c_aa = self.XYZData._wrap_molecules( self.mol_map, self.cell_aa_deg )

    def install_molecular_origin_gauge( self, **kwargs ):
        '''Script mainly from Arne Scherrer'''
        fn = kwargs.get('fn_topo')
        if fn: #use pdbReader
            try: _fn[ fn ]
            except KeyError: _fn[ fn ] = pdbReader( fn )
            data, types, symbols, residues, box_aa_deg, title = _fn[ fn ]
            resi = ['-'.join(_r) for _r in residues]
            _map_dict = dict( zip( list( dict.fromkeys( resi ) ), range( len( set( resi ) ) ) ) ) #python>=3.6: keeps order
            n_map = [ _map_dict[ _r ] for _r in resi ]
            #n_map = residues[:,0].astype(int).tolist() #-1 as workaround

            #changed 2019-05-23
            #setattr(self,'cell_aa_deg',kwargs.get('cell_aa',box_aa_deg))
            setattr( self, 'cell_aa_deg', box_aa_deg )

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
       #
       #disabled 2019-05-22                                                                                #
       # setattr(self.XYZData,'mol_map',n_map)
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

    def sort_atoms(self, **kwargs):
        '''Sort atoms alphabetically (default)'''
        ind = self.XYZData._sort()
        self.mol_map = self.mol_map[ind]
        #what else?

    def write( self, fn, **kwargs ):
        '''Work in progress...'''
        fmt  = kwargs.get( 'fmt', fn.split( '.' )[ -1 ] )
        nargs = {}
        if fmt == 'pdb':
            nargs = { _s : getattr(self, _s) for _s in ('mol_map', 'cell_aa_deg') }
        else:
            raise NotImplementedError( "System object supports only PDB output for now (use _XYZ attribute instead)" )
        self.XYZData.write( fn, fmt=fmt, **nargs )


class Supercell( _SYSTEM ):
    def _XYZ( self, *args, **kwargs ):
        return XYZTrajectory( *args, **kwargs )

class Molecule( _SYSTEM ):
    def _XYZ( self, *args, **kwargs ):
        return XYZFrame( *args, **kwargs )
