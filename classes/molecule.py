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
class Molecule():
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

class XYZData(): #later: merge it with itertools (do not load any traj data before the actual processing)        
    def __init__(self, *args, **kwargs): #**kwargs for named (dict), *args for unnamed
        self.axis_pointer = kwargs.get( 'axis_pointer', 0 )
        align_coords = kwargs.get( 'align_atoms', False )
        center_coords = kwargs.get( 'center_coords', False )
        if len( args ) > 1: raise TypeError( "%s takes at most 1 argument!" % self.__class__.__name__ )
        elif len( args ) == 1: 
            fn = args[ 0 ]
            fmt = kwargs.get( 'fmt' , fn.split( '.' )[ -1 ] )
            if fmt == "xyz":        
                self.fn = fn #later: read multiple files
                data, symbols, comments = xyzReader( fn )
            elif fmt=="xvibs":
                self.fn = fn 
                comments = [ "xvibs" ]
                n_atoms, numbers, coords_aa, n_modes, omega_invcm, modes = xvibsReader( fn )
                symbols  = [ constants.symbols[ z - 1 ] for z in numbers ]
                data     = coords_aa.reshape( ( 1, n_atoms, 3 ) )
            else:
                raise Exception( 'Unknown format: %s.' % fmt )

        elif len( args ) == 0:
            #if all( _a in kwargs for _a in [ 'data', 'symbols' ] ): 
            if 'data' in kwargs and ( 'symbols' in kwargs or 'numbers' in kwargs ):
                self.fn = '' 
                comments = kwargs.get( 'comments', [ 'passed' ] )
                numbers = kwargs.get( 'numbers' )
                symbols  = kwargs.get( 'symbols' )
                if symbols is None: symbols = [ constants.symbols[ z - 1 ] for z in numbers ]
                data     = kwargs.get( 'data' )
                _sh = data.shape
                if len( _sh ) == 2: 
                    data = data.reshape( ( 1, ) + _sh )
                
            else: raise TypeError( 'XYZData needs fn or data + symbols argument!' )


        self.comments = np.array(comments)            
        self.symbols  = np.array(symbols)
        self.data     = data #should be replaced by pos_aa
        self.pos_aa   = data[:,:,:3]
        self.vel_au   = data[:,:,3:]
        self._sync_class()

        if align_coords: #extra function?
            print('Aligning atoms.')
            self.pos_aa = align_atoms(self.pos_aa,self.masses_amu,ref=self.pos_aa[0])

        if center_coords: #extra function?
            print('Centering coordinates in cell and wrap atoms.')
            cell_aa = kwargs.get('cell_aa',np.array([0.0,0.0,0.0,90.,90.,90.]))
            if not all([cl == 90. for cl in cell_aa[3:]]): 
                print([cl for cl in cell_aa[3:]])
                print('ERROR: only orthorhombic/cubic cells can be used with center function!')
                sys.exit(1)
            P = self.pos_aa[:,:,:3]
            M = self.masses_amu
            com_aa = np.sum(P*M[None,:,None], axis=-2)/M.sum()
            self.pos_aa[:,:,:3] += cell_aa[None,None,:3]/2 - com_aa[:,None,:]
            #wrap as option?
            print('WARNING: Auto-wrap of atoms activated!')
            if not any(_c==0.0 for _c in cell_aa[:3]): self.pos_aa = np.remainder(self.pos_aa,cell_aa[:3])

        self._sync_class()

            
    def _sync_class(self):
        try:
            self.masses_amu = np.array([masses_amu[s] for s in self.symbols])
        except  KeyError:
            print('WARNING: Could not find all element masses!')
        if self.vel_au.size == 0: self.vel_au = np.zeros(self.pos_aa.shape)
        self.n_frames,self.n_atoms,self.n_fields = self.pos_aa.shape
                   
    def __add__(self,other):
        new = copy.deepcopy(self)
        new.pos_aa = np.concatenate((self.pos_aa,other.pos_aa),axis=self.axis_pointer)
        new._sync_class()
        if self.axis_pointer == 0:
            self.comments = np.concatenate((self.comments,other.comments))        
        if new.axis_pointer == 1:
            new.symbols = np.concatenate((self.symbols,other.symbols)) #not so beautiful
        return new
    
    def __iadd__(self,other):
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
    
    def _sort(self): #NEW
        elem = {s:np.where(self.symbols==s)[0] for s in np.unique(self.symbols)}
        ind = [i for k in sorted(elem) for i in elem[k]]
        self.pos_aa = self.pos_aa[:,ind,:]
        self.vel_au = self.vel_au[:,ind,:]
        self.symbols = self.symbols[ind]
        self._sync_class()
    
    def write(self,fn,**kwargs):
        if fn == self.fn:
            raise Exception('ERROR: write file equals source file. File would be overwritten!')
        attr = kwargs.get('attr','pos_aa')
        factor = kwargs.get('factor',1.0) #for velocities
        separate_files = kwargs.get('separate_files',False) #only for XYZ

        if attr=='data': self.data = np.concatenate((self.pos_aa,self.vel_au),axis=-1) #TMP solution; NB: CPMD writes XYZ files with vel_aa 
        fmt  = kwargs.get('fmt',fn.split('.')[-1])
        if fmt=="xyz":
            if separate_files: 
                frame_list = kwargs.get('frames',range(self.n_frames))
                [xyzWriter(''.join(fn.split('.')[:-1])+'%03d'%fr+'.'+fn.split('.')[-1],
                             [getattr(self,attr)[fr]],self.symbols,[self.comments[fr]]) for fr in frame_list]
          
            else: xyzWriter(fn,
                            getattr(self,attr),
                            self.symbols,
                            self.comments)    #Writer is stupid
                            #self.n_frames*self.comments)    #Writer is stupid
        elif fmt=="pdb":
            for _attr in ['mol_map','abc','albega']: #try to conceive missing data from kwargs
                try: getattr(self,_attr)
                except AttributeError: setattr(self,_attr,kwargs.get(_attr))
            pdbWriter(fn,
                      self.pos_aa[0], #only frame 0 vels are not written
                      types=self.symbols,#if there are types change script
                      symbols=self.symbols,
                      residues=np.vstack((np.array(self.mol_map)+1,np.array(['MOL']*self.symbols.shape[0]))).swapaxes(0,1), #+1 because no 0 index
                      box=np.hstack((self.abc,self.albega)),
                      title='Generated from %s with Molecule Class'%self.fn)

        elif fmt == 'cpmd': #pos and vel, attr does not apply
            print('CPMD WARNING: Output with sorted atomlist!')
            loc_self = copy.deepcopy(self)
            loc_self._sort()
            cpmdWriter(fn, loc_self.pos_aa*Angstrom2Bohr, loc_self.symbols, loc_self.vel_au*factor, **kwargs) # DEFAULTS pp='MT_BLYP', bs=''

        else: raise Exception('Unknown format: %s.'%fmt)

    def _is_similar(self,other): #add more tests later
        #used by methods:
        #topology.map_atoms_by_coordinates
        f = lambda a: getattr(self,a)==getattr(other,a)
        ie = list(map(f,['n_atoms','n_fields']))
        ie.append(bool(np.prod([a==b for a,b in zip(np.sort(self.symbols),np.sort(other.symbols))])))
        #if hasattr(data,'cell_aa')
        return np.prod(ie),ie

    #### The next two methods should be externalised
    def _move_residue_to_centre(self,ref,cell_aa_deg,**kwargs):
       try: ref_pos_aa = getattr(self,'mol_cog_aa')[:,ref]
       except: ref_pos_aa = getattr(self,'mol_com_aa')[:,ref]
       if not all([cl == 90. for cl in cell_aa_deg[3:]]): 
           print([cl for cl in cell_aa_deg[3:]])
           print('ERROR: only orthorhombic/cubic cells can be used with center function!')
           sys.exit(1)
       P = self.pos_aa[:,:,:3]
       self.pos_aa[:,:,:3] += cell_aa_deg[None,None,:3]/2 - ref_pos_aa[:,None,:]

    def _wrap_molecules(self,mol_map,cell_aa_deg,**kwargs): #another routine would be complete_molecules for both-sided completion
        mode = kwargs.get('mode','cog')
        abc,albega = np.split(cell_aa_deg,2)
        if not np.allclose(albega,np.ones((3))*90.0):
            raise Exception('ERROR: Only orthorhombic cells implemented for mol wrap!')
        pos_aa = np.array([dec(self.pos_aa[fr,:,:3], mol_map) for fr in range(self.n_frames)])
        w = np.ones((self.n_atoms))
        if mode=='com': w = self.masses_amu
        w = dec(w, mol_map)
        mol_c_aa = []
        cowt = lambda x,wt: np.sum(p*wt[None,:,None], axis=1)/wt.sum()
        for i_mol in range(max(mol_map)+1):
            ind = np.array(mol_map)==i_mol
            p = self.pos_aa[:,ind,:3]
            if not any([_a<=0.0 for _a in abc]):
                p -= np.around((p-p[:,0,None,:])/abc[None,None,:])*abc[None,None,:] 
                c_aa = cowt(p,w[i_mol])
                mol_c_aa.append(np.remainder(c_aa,abc[None,:])) #only for orthorhombic cells
            else: 
                print('WARNING: Cell size zero!')
                c_aa = cowt(p,w[i_mol])
                mol_c_aa.append(c_aa)

            self.pos_aa[:,ind,:3] = p-(c_aa-mol_c_aa[-1])[:,None,:]
        print('UPDATE WARNING: inserted "swapaxes(0,1)" for mol_cog_aa attribute (new shape: (n_frames,n_mols,3))!')
        setattr(self,'mol_'+mode+'_aa',np.array(mol_c_aa).swapaxes(0,1))
        setattr(self,'mol_map',mol_map)
        setattr(self,'abc',abc)
        setattr(self,'albega',albega)



    def calculate_nuclear_velocities(self,**kwargs): #finite diff, linear (frame1-frame0, frame2-frame1, etc.)
        temperature = kwargs.get('temperature',300)
        ts = kwargs.get('ts',0.5)

        if np.linalg.norm(self.vel_au) != 0: print('WARNING: Overwriting existing velocities in file %s'%self.fn)
        self.vel_au[:-1] = np.diff(self.pos_aa,axis=0)/(ts*constants.v_au2aaperfs)#*np.sqrt(self.masses_amu)[None,:,None]

#        norm = np.linalg.norm(vec,axis=(1,2)) 
#        vec /= norm[:,None,None] # treat as normal mode
        # Adapted from Arne Scherrer. Occupation can be single, average, or random.
#        beta_au = 1./(temperature*constants.k_B_au)
#        S = vec/np.sqrt(beta_au)#/np.sqrt(constants.m_amu_au)/np.sqrt(self.masses_amu)[None,:,None]  
        # occupation = 'single'
#        self.vel_au[:-1] = S #last frame remains unchanged
#        e_kin_au = CalculateKineticEnergies(self.vel_au,self.masses_amu)
#        scale = temperature/(np.sum(e_kin_au)/constants.k_B_au/self.n_frames-1)/2



class VibrationalModes():
    #actually here and for the entire MD simulation we should choose one pure isotope since mixed atomic masses are not valid for the determination for "average" mode frequencies (are they?)
    #allow partial reading of modes, insert check for completness of modes 3N-6/5
    def __init__(self,fn,**kwargs): #**kwargs for named (dict), *args for unnamed
        fmt = kwargs.get('fmt',fn.split('.')[-1])
        center_coords = kwargs.get('center_coords',False)
        if 'modes' in kwargs and 'numbers' in kwargs and 'omega_cgs' in kwargs and 'coords_aa' in kwargs:
            self.fn = fn 
            modes   = kwargs.get('modes')
            numbers = kwargs.get('numbers')
            omega_cgs = kwargs.get('omega_cgs')
            coords_aa = kwargs.get('coords_aa')
            comments = ["passed"]
            symbols  = [constants.symbols[z-1] for z in numbers]
            pos_au   = coords_aa*constants.l_aa2au
            eival_cgs = omega_cgs
        elif fmt=="xvibs":
            self.fn = fn
            comments = ["xvibs"]
            n_atoms, numbers, coords_aa, n_modes, omega_cgs, modes = xvibsReader(fn)
            symbols  = [constants.symbols[z-1] for z in numbers]
            pos_au   = coords_aa*constants.l_aa2au
            eival_cgs = omega_cgs
            
#        elif fmt=="molvib": #mass weighted hessian as used in CPMD
#            self.fn = fn 
        else:
            raise Exception('Unknown format: %s.'%fmt)


        self.pos_au = pos_au
        self.comments = np.array(comments)
        self.symbols  = np.array(symbols)
        self.masses_amu = np.array([masses_amu[s] for s in self.symbols])
        self.eival_cgs = np.array(eival_cgs)
        self.modes = modes
        #new_eivec = modes*np.sqrt(self.masses_amu*constants.m_amu_au)[None,:,None]
        new_eivec = modes*np.sqrt(self.masses_amu)[None,:,None]
        self.eivec  = new_eivec#/np.linalg.norm(new_eivec,axis=(1,2))[:,None,None]  #usually modes have been normalized after mass-weighing, so eivecs have to be normalized again 
        self._sync_class()

        if center_coords:
            cell_aa = kwargs.get('cell_aa',np.array([0.0,0.0,0.0,90.,90.,90.]))
            if not all([cl == 90. for cl in cell_aa[3:]]): 
                print('ERROR: only orthorhombic/cubic cells can be used with center function!')
            P = self.pos_au
            M = self.masses_amu
            com_au = np.sum(P*M[:,None], axis=-2)/M.sum()
            self.pos_au += cell_aa[None,:3]/2*constants.l_aa2au - com_au[None,:]

        self._sync_class()
        self._check_orthonormality()

    def _check_orthonormality(self):
        atol=5.E-5
        com_motion = np.linalg.norm((self.modes*self.masses_amu[None,:,None]).sum(axis=1),axis=-1)/self.masses_amu.sum()
        if np.amax(com_motion) > atol: print('WARNING: Significant motion of COM for certain modes!')
        test = self.modes.reshape(self.n_modes,self.n_atoms*3)
        a=np.inner(test,test)
        if any([np.allclose(a,np.identity(self.n_modes),atol=atol),np.allclose(a[6:,6:],np.identity(self.n_modes-6),atol=atol)]): 
            print('ERROR: The given cartesian displacements are orthonormal! Please try to enable/disable the -mw flag!')
            if not ignore_warnings:
                sys.exit(1)
            else:
                print('IGNORED')
        test = self.eivec.reshape(self.n_modes,self.n_atoms*3)
        a=np.inner(test,test)
        if not any([np.allclose(a,np.identity(self.n_modes),atol=atol),np.allclose(a[6:,6:],np.identity(self.n_modes-6),atol=atol)]): 
            print(a)
            print('ERROR: The eigenvectors are not orthonormal!')
            print(np.amax(np.abs(a-np.identity(self.n_modes))))
            if not ignore_warnings:
                sys.exit(1)
            else:
                print('IGNORED')

    def _check_orthonormality_OLD_WITH_ROT_TRANS(self):
        atol=5.E-5
        com_motion = np.linalg.norm((self.modes*self.masses_amu[None,:,None]).sum(axis=1),axis=-1)/self.masses_amu.sum()
        if np.amax(com_motion) > atol: print('WARNING: Significant motion of COM for certain modes!')
        test = self.modes.reshape(self.n_modes,self.n_atoms*3)
        a=np.inner(test,test)[6:,6:]
        if np.allclose(a,np.identity(self.n_modes-6),atol=atol): raise Exception('ERROR: The given cartesian displacements are orthonormal! Please try to enable/disable the -mw flag!')
        test = self.eivec.reshape(self.n_modes,self.n_atoms*3)
        a=np.inner(test,test)[6:,6:]
        if not np.allclose(a,np.identity(self.n_modes-6),atol=atol): 
            print(a)
            print('ERROR: The eigenvectors are not orthonormal!')
            print(np.amax(np.abs(a-np.identity(self.n_modes-6))))
            if not ignore_warnings:
                sys.exit(1)
            else:
                print('IGNORED')

    def _sync_class(self):
        self.masses_amu = np.array([masses_amu[s] for s in self.symbols])
        self.n_modes,self.n_atoms,three = self.modes.shape
        norm = np.linalg.norm(self.eivec,axis=(1,2)) #usually modes have been normalized after mass-weighing, so eivecs have to be normalized again
        norm[:6] = 1.0 #trans+rot
        self.eivec /= norm[:,None,None] 

    def __add__(self,other):
        new = copy.deepcopy(self)
        new.pos_au = np.concatenate((self.pos_au,other.pos_au),axis=0)
        new.symbols = np.concatenate((self.symbols,other.symbols)) #not so beautiful
        new.eivec = np.concatenate((self.eivec,other.eivec),axis=1) #axis 0 are the modes
        new.modes = np.concatenate((self.modes,other.modes),axis=1) #axis 0 are the modes
        new._sync_class()
        return new
    
    def __iadd__(self,other):
        self.pos_au = np.concatenate((self.pos_au,other.pos_au),axis=0)
        self.symbols = np.concatenate((self.symbols,other.symbols))            
        self.eivec = np.concatenate((self.eivec,other.eivec),axis=1) #axis 0 are the modes
        self.modes = np.concatenate((self.modes,other.modes),axis=1) #axis 0 are the modes
        self._sync_class()
        return self

    def _sort(self): #NEW
        elem = {s:np.where(self.symbols==s)[0] for s in np.unique(self.symbols)}
        ind = [i for k in sorted(elem) for i in elem[k]]
        self.pos_au = self.pos_au[ind,:]
        self.symbols = self.symbols[ind]
        self.masses_amu = self.masses_amu[ind]
        self.eivec = self.eivec[:,ind,:]
        self.modes = self.modes[:,ind,:]

        if hasattr(self,'vel_au'): self.vel_au = self.vel_au[ind,:]
        self._sync_class()

    def get_transition_moments(self,source,**kwargs): # adapted from Arne Scherrer 
        def dec(prop, indices):
            """decompose prop according to indices"""
            return [np.array([prop[k] for k, j_mol in enumerate(indices) if j_mol == i_mol]) for i_mol in range(max(indices)+1)]

        if source=='cpmd_nvpt_md': #all data in a.u.
            '''adapted from Arne Scherrer'''
        #modelist contains all modes of the class that are to be considered. Each mode corresponds to one frame in traj and moms file
            fn_traj = kwargs.get('fn_traj')
            fn_moms = kwargs.get('fn_moms')
            modelist = kwargs.get('modelist',range(self.n_modes))
            if fn_traj==None or fn_moms==None: raise Exception('ERROR: Please give fn_traj and fn_moms for source "cpmd_nvpt_md"!')
            cell_au=getattr(self,'cell_au',None)

            ZV = np.array([valence_charges[s] for s in self.symbols])
            n_atoms, n_moms = len(self.symbols), sum(ZV)//2 #all electron calcs? -> do not use ZV but Z

            self.n_states = n_moms
            self.c_au = np.zeros((self.n_modes,3))
            self.m_au = np.zeros((self.n_modes,3))
            self._transport_term_au = np.zeros((self.n_modes,3))
#            self.el_c_au = np.zeros((self.n_modes,3))
#            self.nu_c_au = np.zeros((self.n_modes,3))
#            self.el_m_au = np.zeros((self.n_modes,3))
#            self.nu_m_au = np.zeros((self.n_modes,3))
            self._r_wc_au = np.zeros((self.n_modes,self.n_states,3))
            self._sw_c_au = np.zeros((self.n_modes,self.n_states,3))
            self._sw_m_dwc_au = np.zeros((self.n_modes,self.n_states,3))

            if hasattr(self,'mol_map'):
                print('Using molecular gauge.')
                coms = self.mol_com_au
                n_map = self.mol_map
                self.mol_c_au = np.zeros((self.n_modes,self.n_mols,3))
                self.mol_m_au = np.zeros((self.n_modes,self.n_mols,3))
            else:
                coms = (np.sum(self.pos_au*self.masses_amu[:,None], axis=0)/self.masses_amu.sum()).reshape((1,3))
                n_map = tuple(np.zeros((self.n_atoms)).astype(int))

            ZV = dec(ZV, n_map)

            for i_mode, (pos, vel, wc, c, m) in enumerate(cpmd_n.get_frame_traj_and_mom(fn_traj,fn_moms,n_atoms,n_moms)):
              if i_mode >= len(modelist): #NEW: if modelist has less entries than trajectory
                print('WARNING: Trajectory file contains more entries than given modelist!')
                break
              else:
                if not np.allclose(self.pos_au,pos): 
                    test=np.unique(np.around(pos-self.pos_au,6))
                    if test.shape==(3,):
                        print('WARNING: fn_traj coordinates shifted by vector %s with respect to stored coordinates!'%test)
                    else:
                        print('ERROR: fn_traj not consistent with nuclear coordinates!')#,np.around(pos-self.pos_au,6))
                        if not ignore_warnings:
                            sys.exit(1)
                        else:
                            print('IGNORED')

                #raw data (for mtm)
                self._sw_c_au[modelist[i_mode]] = c
                self._sw_m_dwc_au[modelist[i_mode]] = m
                self._r_wc_au[modelist[i_mode]] = wc

                # assign Wannier centers to molecules
                dists = pos[np.newaxis,:,:] - wc[:,np.newaxis,:]
                if hasattr(cell_au,'shape'):
                    dists -= np.around(dists/cell_au)*cell_au
                e_map = [n_map[np.argmin(state_distances)] for state_distances in np.sum(dists**2, axis=2)]
                # decompose data into molecular contributions
                pos, vel = dec(pos, n_map), dec(vel, n_map)
                wc, c, m = dec(wc, e_map), dec(c, e_map), dec(m, e_map)

                mol_c,mol_m = list(),list()
                for i_mol,com in enumerate(coms): #wannier2molecules
                # calculate nuclear contributions to molecular moments and the molecular current dipole moment
                    el_c = c[i_mol]
                    el_m = m[i_mol].sum(axis=0)
                    el_m += magnetic_dipole_shift_origin(wc[i_mol],el_c,origin_au=com,cell_au=cell_au)

                    nu_c = current_dipole_moment(vel[i_mol],ZV[i_mol]) 
                    nu_m = np.zeros(nu_c.shape).sum(axis=0)
                    nu_m += magnetic_dipole_shift_origin(pos[i_mol],nu_c,origin_au=com,cell_au=cell_au)

                    mol_c.append(nu_c.sum(axis=0)+el_c.sum(axis=0))
                    mol_m.append(nu_m + el_m)

                mol_c=np.array(mol_c)
                mol_m=np.array(mol_m)

                for i_mol,com in enumerate(coms): #molecules2common no scaling
                    self.c_au[modelist[i_mode]] += mol_c.sum(axis=0)
                    self.m_au[modelist[i_mode]] += mol_m.sum(axis=0) + magnetic_dipole_shift_origin(coms,mol_c,origin_au=com,cell_au=cell_au)
                    self._transport_term_au[modelist[i_mode]] += magnetic_dipole_shift_origin(coms,mol_c,origin_au=com,cell_au=cell_au)
                    #self.m_au[modelist[i_mode]] +=  magnetic_dipole_shift_origin(coms,mol_c,origin_au=com,cell_au=cell_au)
                    #self.el_c_au[modelist[i_mode]] += el_c.sum(axis=0)
                    #self.nu_c_au[modelist[i_mode]] += nu_c.sum(axis=0)
                    #self.el_m_au[modelist[i_mode]] += el_m
                    #self.nu_m_au[modelist[i_mode]] += nu_m

                if hasattr(self,'mol_map'): 
                    self.mol_c_au[modelist[i_mode]] = mol_c
                    self.mol_m_au[modelist[i_mode]] = mol_m
            if not i_mode+1 == self.n_modes: print('WARNING: Did not find data for all modes. Read only %d modes.'%i_mode)

        elif source=='cpmd_nvpt_at':
            ### import or explicitly exclude mol handling from ipython notebook #22
            fn_APT = kwargs.get('fn_APT')
            fn_AAT = kwargs.get('fn_AAT')
            if fn_APT==None or fn_AAT==None: raise Exception('ERROR: Please give fn_APT and fn_AAT for source "cpmd_nvpt_at"!')
            self.c_au = np.zeros((self.n_modes,3))
            self.m_au = np.zeros((self.n_modes,3))
            
            self.APT = np.loadtxt(fn_APT).astype(float).reshape(self.n_atoms, 3, 3)
            self.AAT = np.loadtxt(fn_AAT).astype(float).reshape(self.n_atoms, 3, 3)
            sumrule = constants.e_si**2*constants.avog*np.pi*np.sum(self.APT**2/self.masses_amu[:,np.newaxis,np.newaxis])/(3*constants.c_si**2)/constants.m_amu_si
            print(sumrule)
            #modes means cartesian displacements
            self.c_au = (self.modes[:,:,:,np.newaxis]*self.APT[np.newaxis,:,:,:]).sum(axis=2).sum(axis=1)
            self.m_au = (self.modes[:,:,:,np.newaxis]*self.AAT[np.newaxis,:,:,:]).sum(axis=2).sum(axis=1)
            #INSERT HERE MOLECULAR GAUGE

            
        else: # orca, ...
            raise Exception('Unknown or unimplemented source: %s.'%source)

    def mtm_calculate_mic_contribution(self,box_vec_aa,source,**kwargs):
        '''results is origin-independent'''
        if source=='cpmd_nvpt_md': #all data in a.u.
        #modelist contains all modes of the class that are to be considered. Each mode corresponds to one frame in cpmd file
            fn_e0 = kwargs.get('fn_e0')
            fn_r1 = kwargs.get('fn_r1')
            modelist = kwargs.get('modelist')
            if fn_e0==None or fn_r1==None or modelist==None: raise Exception('ERROR: Please give fn_e0, fn_r1, and modelist for source "cpmd_nvpt_md"!')
            E0,R1 = cpmd_n.extract_mtm_data_tmp(fn_e0,fn_r1,len(modelist),self.n_states)
            com_au = np.sum(self.pos_au*self.masses_amu[:,None], axis=0)/self.masses_amu.sum()
        #    if origin == 'center':
            #origin_aa = box_vec_aa/2
            origin_aa = np.zeros(box_vec_aa.shape)
            self.m_ic_r_au = np.zeros((self.n_modes,3))
            self.m_ic_t_au = np.zeros((self.n_modes,3))
            for im,mode in enumerate(modelist): 
                r_aa = (self._r_wc_au[mode]-com_au[np.newaxis,:])*constants.l_au2aa+origin_aa[np.newaxis,:]
                r_aa -= np.around(r_aa/box_vec_aa)*box_vec_aa
                m_ic_r,m_ic_t = calculate_mic(E0[im],R1[im],self._sw_c_au[mode],self.n_states,r_aa,box_vec_aa)
                self.m_ic_r_au[mode] = m_ic_r
                self.m_ic_t_au[mode] = m_ic_t
            self.m_lc_au = copy.deepcopy(self.m_au)
            self.m_au += self.m_ic_r_au#+self.m_ic_t_au
        
    def calculate_mtm_spectrum(self):
        self.m_au = copy.deepcopy(self.m_lc_au)
        self.calculate_spectrum()
        self.continuous_spectrum(self.n_modes*[1])
        self.D_cgs_lc = copy.deepcopy(self.D_cgs)
        self.R_cgs_lc = copy.deepcopy(self.R_cgs)
        self.ira_spec_lc = copy.deepcopy(self.ira_spec)
        self.vcd_spec_lc = copy.deepcopy(self.vcd_spec)
        self.m_au += self.m_ic_r_au#+self.m_ic_t_au
        self.calculate_spectrum()
        self.continuous_spectrum(self.n_modes*[1])

    def load_localised_power_spectrum(self,fn_spec):
        a=np.loadtxt(fn_spec)
        self.nu_cgs = a[:,0] #unit already cm-1 if coming from molsim
        self.pow_loc = a[:,1:].swapaxes(0,1) #unit?

    def calculate_spectral_intensities(self): #SHOULDN'T BE METHOD OF CLASS
        self.D_cgs = (self.c_au*self.c_au).sum(axis=1) #going to be cgs
        self.R_cgs = (self.c_au*self.m_au).sum(axis=1)
    
        #rot_str_p_p_trans = np.zeros(n_modes)
        #rot_str_p_p_diff  = np.zeros(n_modes)
        # NEW STUFF
        #    dip_intensity      = dip_str*IR_int_kmpmol
        #    dip_str           *= dip_str_cgs/omega_invcm
        #    rot_str_m_p       *= rot_str_cgs #**2
        
        #Understand units later
        atomic_mass_unit = constants.m_amu_au #1822.88848367
        ev2au = 1/np.sqrt(atomic_mass_unit)  # eigenvalue to atomic units
        ev2wn = ev2au/(2*np.pi*constants.t_au*constants.c_si)/100 # eigenvalues to wave numbers
        au2wn = 1/(2*np.pi*constants.t_au*constants.c_si)/100 # atomic units to wave numbers
        ###################################################################################################
        # Dipole Strength
        # D_n = APT^2 S_n^2  hbar/(2 omega_n)
        # [S_n^2]     = M^-1  -> (m_u' m_e)^-1
        # [APT^2]     = Q^2   -> e^2
        # [hbar/omega_n] = M L^2 -> m_e l_au^2  (via au2wn/omega)
        # altogether this yields e^2 l_au^2 / m_u' -> needs to be converted to cgs
        e_cgs    = constants.e_si*constants.c_si*1E1 # 4.80320427E-10 # use constants.e_cgs ?
        l_au_cgs = constants.l_au*1E2
        dip_str_cgs = (e_cgs*l_au_cgs*ev2au)**2*au2wn*1E40/2 # convert 1/omega back to au
        ###################################################################################################
        # Rotational Strength
        # R_n = 1/c AAT APT S_n^2 omega_n hbar/(2 omega_n) = hbar/(2 c) AAT APT S_n^2
        # [S_n^2]     = M^-1  -> (m_u' m_e)^-1
        # [APT]       = Q     -> e
        # [AAT]       = Q L   -> e l_au (NB! c_au is not yet included!)
        # altogether this yields hbar e^2 l_au/(2 c m_e) (/m_u'?!)
        c_cgs = constants.c_si*1E2
        m_e_cgs = constants.m_e_si*1E3
        hbar_cgs = constants.hbar_si*1E7
        # rot_str_cgs = (e_cgs*l_au_cgs*ev2au)**2/constants.c_au*1E44/2 # alpha = ...
        rot_str_cgs = (hbar_cgs*l_au_cgs*(e_cgs*ev2au)**2)/(c_cgs*m_e_cgs)*1E44/2
        ###################################################################################################
        # IR Intensity in km/mol-1
        # prefactor (compare master A.97 or Eq. (13) in J. Chem. Phys. 134, 084302 (2011)): Na beta/6 epsilon0 c 
        # harmonic approximation A.98-A.99 yields a delta function*pi(*2?) ... beta^-1 = hbar omega / 2 (A.89-A.91)
        # conversion to wavenumbers yields nu = omega/2 pi c a factor of 1/2 pi c ([delta(x)] = 1/[x])
        # APTs give (e**2/m_p)
        # conversion in km**-1 gives 1E-3
        # factor 1/3 is due to averaging
        # altogether this yields
        # (N_a e**2)/(6*epsilon0 c m_p)*1E-3
        epsilon0_si=8.85418782000E-12 #10(-12)C2.N-1.m-2
        IR_int_kmpmol=(constants.avog*constants.e_si**2)/(12*epsilon0_si*constants.c_si**2*constants.m_p_si*1000)
    
        IR_int_kmpmol=(constants.avog*constants.e_si**2)/(12*epsilon0_si*constants.c_si**2*constants.m_p_si*1000)
        dip_str_cgs = (e_cgs*l_au_cgs*ev2au)**2*au2wn*1E40/2 # convert 1/omega back to au
        rot_str_cgs = (hbar_cgs*l_au_cgs*(e_cgs*ev2au)**2)/(c_cgs*m_e_cgs)*1E44/2
    
        self.I_kmpmol = self.D_cgs*IR_int_kmpmol #here, D_cgs is still au!
        self.D_cgs   *= dip_str_cgs/self.eival_cgs
        self.R_cgs   *= rot_str_cgs
        #rot_str_p_p_trans *= rot_str_cgs/2
        #rot_str_p_p_diff  *= rot_str_cgs/4
        #VCD intensity?

#    def discrete_spectrum():
        
    def continuous_spectrum(self,widths=None, nu_min_cgs=0, nu_max_cgs=3800, d_nu_cgs=2):
        def Lorentzian1(x, width, height, position):
            numerator =  1
            denominator = (x-position)**2 + width**2
            y = height*(numerator/denominator)/np.pi
            return y
        def Lorentzian2(x, width, height, position):
            numerator =  width
            denominator = (x-position)**2 + width**2
            y = height*(numerator/denominator)/np.pi
            return y
        
        try:
            n_points      = self.nu_cgs.shape[0]
            print('Found already loaded spectral data. Using now its frequency range.')
        except AttributeError:
            self.nu_cgs = np.arange(nu_min_cgs, nu_max_cgs, d_nu_cgs)
            n_points      = self.nu_cgs.shape[0]

        self.ira_spec = np.zeros((1+self.n_modes, n_points))
        self.vcd_spec = np.zeros((1+self.n_modes, n_points))

        D_cgs = self.D_cgs*1E-40 #[esu2cm2]
        R_cgs = self.R_cgs*1E-44 #[esu2cm2]
        prefactor_cgs = (4*np.pi**2*constants.avog)/(3*constants.c_cgs*constants.hbar_cgs)
        cm2_to_kmcm   = 1E-5
        D_scaled = D_cgs*1*prefactor_cgs*cm2_to_kmcm
        R_scaled = R_cgs*4*prefactor_cgs*cm2_to_kmcm

        try:
            for k, nu_k in enumerate(self.eival_cgs):
                self.ira_spec[k+1,:] = self.pow_loc[k,:]*D_scaled[k]*nu_k #units?
                self.vcd_spec[k+1,:] = self.pow_loc[k,:]*R_scaled[k]*nu_k #units?
                self.ira_spec[0,:]  += self.ira_spec[k+1,:]
                self.vcd_spec[0,:]  += self.vcd_spec[k+1,:]
            print('Found localised spectral data. I will use it as basis for ira and vcd calculation (instead of Lorentzian).')
        except AttributeError:
            if widths==None:
                widths=self.n_modes*[1]
            for k, nu_k in enumerate(self.eival_cgs):
                self.ira_spec[k+1,:] = Lorentzian2(self.nu_cgs, widths[k], D_scaled[k]*nu_k, nu_k)
                self.vcd_spec[k+1,:] = Lorentzian2(self.nu_cgs, widths[k], R_scaled[k]*nu_k, nu_k)
                self.ira_spec[0,:]  += self.ira_spec[k+1,:]
                self.vcd_spec[0,:]  += self.vcd_spec[k+1,:]

    def calculate_nuclear_velocities(self,**kwargs): #The velocities are so small? #SHOULDN'T BE MATHOD OF CLASS
        '''Adapted from Arne Scherrer. Occupation can be single, average, or random.'''
        occupation = kwargs.get('occupation','single')
        temperature = kwargs.get('temperature',300)
        beta_au = 1./(temperature*constants.k_B_au)
        S = self.eivec/np.sqrt(beta_au)/np.sqrt(constants.m_amu_au)/np.sqrt(self.masses_amu)[None,:,None]  #use eivec not modes due to normalisation of the latter (all in all not so happy :( )
        if occupation=='single':
            self.vel_au = S
            e_kin_au = CalculateKineticEnergies(self.vel_au,self.masses_amu)
            scale = temperature/(np.sum(e_kin_au)/constants.k_B_au/self.n_modes)/2
        elif occupation=='average':
            self.vel_au = S.sum(axis=0)
            # atomic_ekin_au = traj_utils.CalculateKineticEnergies(avg, masses_amu)
            # scale = temperature/(np.sum(atomic_ekin_au)/constants.k_B_au/n_modes)/2
            # print(scale)
            # avg *= np.sqrt(scale)
        elif occupation=='random': #NOT TESTED!
            phases = np.random.rand(self.n_modes)*np.pi
            self.vel_au = (S*np.cos(phases)[:,None,None]).sum(axis=0)

            #random pos
#            avg = np.zeros((1,n_atoms,3))
#            omega_au = 2*np.pi*np.array(self.eival_cgs)*constants.c_cgs*constants.t_au
#            for i_mode in range(self.n_modes):
#                avg += S[i_mode].reshape(1,self.n_atoms,3)*np.sin(phases[i_mode])/omega_au[i_mode]
#            avg *= constants.l_au2aa
#            avg += self.pos_au*constants.l_au2aa
            print('Random seed not tested')
        else: print('Occupation mode %s not understood!'%occupation)

    def write_nuclear_velocities(self,fn,**kwargs):
        fmt =  kwargs.get('fmt','cpmd') #fn.split('.')[-1])
        modelist = kwargs.get('modelist',range(self.n_modes))
        factor = kwargs.get('factor',1.0)
        loc_n_modes = len(modelist)
        if fmt == 'cpmd': 
            print('CPMD WARNING: Output with sorted atomlist!')
            loc_self = copy.deepcopy(self)
            loc_self._sort()
            pos_au = np.tile(loc_self.pos_au,(loc_self.n_modes,1,1))
            try:
                cpmdWriter(fn, pos_au[modelist], loc_self.symbols, factor*loc_self.vel_au[modelist], offset=0,**kwargs) # DEFAULTS pp='MT_BLYP', bs=''
            except AttributeError:
                loc_self.calculate_nuclear_velocities()
                cpmdWriter(fn, pos_au[modelist], loc_self.symbols, factor*loc_self.vel_au[modelist], offset=0,**kwargs) # DEFAULTS pp='MT_BLYP', bs=''
            del loc_self

        elif fmt=="xyz": #posvel, only vel not implemented
            pos_aa = np.tile(self.pos_au/Angstrom2Bohr,(self.n_modes,1,1))
            try:
                xyzWriter(fn,np.concatenate((pos_aa,factor*self.vel_au),axis=-1)[modelist],self.symbols,[str(m) for m in modelist])
            except AttributeError:
                self.calculate_nuclear_velocities()
                xyzWriter(fn,np.concatenate((pos_aa,factor*self.vel_au),axis=-1)[modelist],self.symbols,[str(m) for m in modelist])

        else: raise Exception('Unknown format: %s'%fmt)
            
    def print_modes(self,fn,**kwargs):
        fmt =  kwargs.get('fmt','xyz') #fn.split('.')[-1])
        modelist = kwargs.get('modelist',range(self.n_modes))
        n_images = kwargs.get('n_images',3)#only odd numbers
        ts_fs = kwargs.get('ts_fs',1)
        loc_n_modes = len(modelist)
        pos_aa = np.tile(self.pos_au*constants.l_au2aa,(n_images,1,1))
        self.calculate_nuclear_velocities()
        img = np.arange(-(n_images//2),n_images//2+1)
        for mode in modelist:
            vel_aa = np.tile(self.vel_au[mode]*constants.t_fs2au*constants.l_au2aa,(n_images,1,1))*img[:,None,None]
            pos_aa += vel_aa*ts_fs
            if fmt=="xyz": 
                xyzWriter('%03d'%mode+'-'+fn,pos_aa,self.symbols,[str(m) for m in img])

#        elif fmt == 'cpmd': #not finished
#            cpmd.WriteTrajectoryFile(fn, pos_aa, self.vel_au[mode], offset=0)

            else: raise Exception('Unknown format: %s'%fmt)

class NormalModes(VibrationalModes):
    #Hessian Things
    pass
     

