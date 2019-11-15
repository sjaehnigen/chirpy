#!/usr/bin/env python
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


import sys
import copy
from collections import OrderedDict

#not yet in use
#from interfaces import cpmd

Angstrom2Bohr = 1.8897261247828971

#this is a method for the CPMD class in interfaces
def _write_atoms_section(fn, symbols, pos_au, pp='MT_BLYP',bs='',fmt='angstrom'):
    '''Only sorted data is comaptible with CPMD, nevertheless auto-sort is disabled.'''
    elems=OrderedDict()
    if pos_au.shape[0] != len(symbols):
        print('ERROR: symbols and positions are not consistent!')
        sys.exit(1)

    pos = copy.deepcopy(pos_au) #copy really necessary? 
    if fmt=='angstrom': pos /= Angstrom2Bohr

    for i,sym in enumerate(symbols):
        if sym != sorted(symbols)[i]: 
            print('WARNING: Atom list not sorted!')
            #sys.exit(1)
        try:
            elems[sym]['n'] +=1
            elems[sym]['c'][elems[sym]['n']] = pos[i]
        except KeyError:
#            if sym in constants.symbols:
#                elems[sym] = constants.species[sym]
            elems[sym] = OrderedDict()
            elems[sym]['n'] = 1
            elems[sym]['c'] = {elems[sym]['n'] : pos[i]}
#            else: raise Exception("Element %s not found!" % sym)

    with open(fn,'w') as f:
        format = '%20.10f'*3 + '\n'
        f.write("&ATOMS\n")
#        f.write(" ISOTOPE\n")
#        for elem in elems:
#            print("  %s" % elems[elem]['MASS'])
        for elem in elems:
            f.write("*%s_%s %s\n" % (elem,pp,bs))
            if elem in ['C','O','N','P','Cl', 'F'] and 'AEC' not in pp:
                f.write(" LMAX=P LOC=P\n")
            else:
                f.write(" LMAX=S LOC=S\n")
            f.write("   %s\n" % elems[elem]['n'])
            for i in elems[elem]['c']:
                f.write(format%tuple([c for c in elems[elem]['c'][i]])) #cp2k format as in pythonbase
        f.write("&END\n")

#This metthod needs to be linked to CPMD class (here only trajectory output, use CPMD class for section output)
def cpmdWriter(fn, pos_au, symbols, vel_au,**kwargs):
    bool_atoms = kwargs.get('write_atoms',True)
    pp = kwargs.get('pp','MT_BLYP')
    bs = kwargs.get('bs','')
    offset = kwargs.get('offset',0)
    '''Adapted from Arne Scherrer'''
    if pos_au.shape[1] != len(symbols):
        print('ERROR: symbols and positions are not consistent!')
        sys.exit(1)

    with open(fn,'w') as f:
        format = ' %16.12f'*3
        for fr in range(pos_au.shape[0]):
            for at in range(pos_au.shape[1]):
                line = '%06d  '%(fr+offset) + format%tuple(pos_au[fr,at])
                line += '  ' + format%tuple(vel_au[fr,at])
                f.write(line+'\n')

    if bool_atoms: 
        _write_atoms_section(fn+'_ATOMS', symbols, pos_au[0], pp=pp, bs=bs)
        xyzWriter(fn+'_ATOMS.xyz',[pos_au[0]/Angstrom2Bohr],symbols,[fn])

#OLD section to be integrated with xyzWriter
def _write_frame(filename, data, symbols, comment, append=False):
    """WriteFrame(filename, data, symbols, comment, append=False)
    Adapted from Arne Scherrer
Input:  
        1. filename: File to read
        2. data: np.array of shape (#atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. comment line (string)
        5. append: Append to file (optional, default = False)

Output: None"""
    format = '  %s'
    # print(data.shape, data.shape[1])
    for field in range(data.shape[1]):
        #format += '    %16.12f'
        format += '      %14.10f' # cp2k format ...
    format += '\n'
    n_atoms = len(symbols)
    obuffer = '% 8d\n'%n_atoms
    # if comment[-1] != '\n':
    #     comment += '\n'
    obuffer += comment.rstrip('\n') + '\n'
    # print(format)
    for i in range(n_atoms):
        tmp = ['{0: <2}'.format(symbols[i])] + [c for c in data[i]]
        obuffer += format%tuple(tmp)
    fmt = 'w'
    if append:
        fmt = 'a'
    with open(filename, fmt) as f:
        f.write(obuffer)


def xyzWriter(fn, data, symbols, comments, append=False):
    """WriteXYZFile(filename, data, symbols, comments, append=False)
    Adapted from Arne Scherrer
Input:  
        1. fn: File to write
        2. data: np.array of shape (#frames, #atoms, #fields/atom)
        3. symbols: list of atom symbols (contains strings)
        4. list of comment lines (contains strings)
        5. append: Append to file (optional, default = False)

Output: None"""
    n_frames = len(comments)
    for fr in range(n_frames):
        _write_frame(fn, data[fr], symbols,
                   comments[fr], append=append or fr != 0)

#from pythonbasei (there is a secon..writers)
def pdbWriter(fn, data, types, symbols, residues, box, title, append=False):
    """WritePDB(fn, data, types, symbols, residues, box, comment, append=False)
Input:  
        1. filename: File to write
        2. data: np.array of shape (#atoms, #fields/atom)
        3. types: list of atom types
        4. symbols: list of atom symbols (contains strings)
        5. residues: np.array of fragment no. consistent to symbols with residue names 
        6. box: list of box parameters (xyz, 3angles)
        7. comment line (string)
        8. append: Append to file (optional, default = False)

Output: None"""

    format = '%s%7d %-5s%-4s%5d    '
    for field in range(data.shape[1]):
        #format += '    %16.12f'
        format += '%8.3f' # cp2k format ...
    format += '%6.2f%6.2f%12s\n'
    n_atoms =  len(symbols)
#    n_frags = 
    obuffer =  'TITLE     %s\n'%title
    obuffer  += 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1%12d\n'%(box[0],box[1],box[2],box[3],box[4],box[5],1)
#    obuffer += comment.rstrip('\n') + '\n'
    for i in range(n_atoms):
        tmp = ['ATOM'] + [i+1] + [types[i]] +  [residues[i][1]] + [int(residues[i][0])] + [c for c in data[i]] + [1] + [0] + [symbols[i]]
#        print(tmp)
        obuffer += format%tuple(tmp)
    obuffer  += 'MASTER        1    0    0    0    0    0    0    0 %4d    0 %4d    0\nEND'%(n_atoms,n_atoms)

    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        f.write(obuffer)

