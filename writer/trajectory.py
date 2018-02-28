#!/usr/bin/env python3

import os
import sys
import numpy as np
import copy
from collections import OrderedDict

Angstrom2Bohr = 1.8897261247828971

def write_atoms_section(fn, symbols, pos_au, pp='MT_BLYP',bs='',fmt='angstrom'):
    '''Only sorted data is comaptible with CPMD, nevertheless auto-sort is disabled.'''
    elems=OrderedDict()
    if pos_au.shape[0] != len(symbols):
        print('ERROR: symbols and positions are not consistent!')
        sys.exit(1)

    pos = copy.deepcopy(pos_au) #copy really necessary? 
    if fmt=='angstrom': pos /= Angstrom2Bohr

    for i,sym in enumerate(symbols):
        if sym != sorted(symbols)[i]: 
            print('ERROR: Atom list not sorted!')
            sys.exit(1)
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

def cpmdWriter(fn, pos_au, symbols, vel_au=None,**kwargs):
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
                if any(vel_au.tolist()): #only for numpy arrays? 
                    line += '  ' + format%tuple(vel_au[fr,at])
                f.write(line+'\n')

    write_atoms_section(fn+'_ATOMS', symbols, pos_au[0], pp=pp, bs=bs)

