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
import numpy as np
from ..physics import constants

#CAUTION: the code below this line has simply been copied from the old pythonbase w/o checks
def myexcepthook(exctype, value, traceback):
    if exctype == Exception:
         print('ERROR:', value)
         #sys.__excepthook__(exctype,value,traceback)

def ReadMoldenVibFile(filename):
    f = open(filename, 'r')
    inbuffer = f.readlines()
    f.close()
    if inbuffer[0].strip() != '[Molden Format]':
        raise Exception('No Molden format?!')
    if inbuffer[1].strip() != '[FREQ]':
        raise Exception('No Frequencies?!')
    freqs = list()
    for line in inbuffer[2:]:
        if '[FR-COORD]' in line:
            break
        else:
            freqs.append(float(line.strip()))
    freqs = np.array(freqs)
    n_modes = freqs.shape[0]
    coords  = list()
    symbols = list()
    for line in inbuffer[n_modes+3:]:
        if '[FR-NORM-COORD]' in line:
            break
        else:
            tmp = line.split()
            symbols.append(tmp[0])
            coords.append([float(e) for e in tmp[1:]])
    coords_aa = np.array(coords)*constants.l_au2aa
    n_atoms = len(symbols)
    vib_data = inbuffer[n_modes+n_atoms+4:]
    modes = np.zeros((n_modes, 3*n_atoms))
    for mode in range(n_modes):
        tmp =  ''.join(vib_data[mode*(n_atoms+1):(mode+1)*(n_atoms+1)][1:])
        modes[mode] = np.array([float(e.strip()) for e in tmp.replace('\n',' ').split()])
    return symbols, coords_aa, freqs, modes


def WriteMoldenVibFile(filename, symbols, coords, freqs, modes):
    n_modes = freqs.shape[0]
    n_atoms = len(symbols)
    format  = '%15f'*3
    format += '\n'
#    modes  = modes_aa*constants.l_aa2au
#    coords = coords_aa*constants.l_aa2au
    f = open(filename, 'w')
    f.write(' [Molden Format]\n [FREQ]\n')
    for i in range(n_modes):
        f.write('%16f\n' % freqs[i])
    f.write(' [FR-COORD]\n')
    for i in range(n_atoms):
        f.write((' %s '+format)%tuple([symbols[i]]+[c for c in coords[i]]))
    f.write(' [FR-NORM-COORD]\n')
    for mode in range(n_modes):
        f.write(' vibration      %i\n' % (mode+1))
        f.write(n_atoms*format%tuple([c for c in modes[mode]]))

# DEPRECATED rsort/msort?
def rsort(ilist):
    sorting, ilist = zip(*sorted(enumerate(ilist),key=lambda x: x[1]))
    return list(ilist), list(sorting)


def msort(ilist,sorting):
    olist = list()
    for i in range(len(ilist)):
        olist.append(ilist[sorting[i]])
    return olist

def ReadAndSortMoldenVibFile(filename):
    f = open(filename, 'r')
    inbuffer = f.readlines()
    f.close()
    if inbuffer[0].strip() != '[Molden Format]':
        sys.excepthook = myexcepthook
        raise Exception('No Molden format?!')
    if inbuffer[1].strip() != '[FREQ]':
        sys.excepthook = myexcepthook
        raise Exception('No Frequencies?!')
    freqs = list()
    for line in inbuffer[2:]:
        if '[FR-COORD]' in line:
            break
        else:
            freqs.append(float(line.strip()))
    freqs = np.array(freqs)
    n_modes = freqs.shape[0]

    atoms   = list()
    coords  = list()
    symbols = list()
    for line in inbuffer[n_modes+3:]:
        if '[FR-NORM-COORD]' in line:
            break
        else:
            atoms.append(line)
    atoms, sorting = rsort(atoms)
    for i,line in enumerate(atoms):
            tmp = line.split()
            symbols.append(tmp[0])
            coords.append([float(e) for e in tmp[1:]])
    coords = np.array(coords)#*constants.l_au2aa
    n_atoms = len(symbols)

    vib_data = inbuffer[n_modes+n_atoms+4:]
    modes = np.zeros((n_modes, 3*n_atoms))
    for mode in range(n_modes):
        tmp =  ''.join(msort(vib_data[mode*(n_atoms+1):(mode+1)*(n_atoms+1)][1:], sorting))
        modes[mode] = np.array([float(e.strip()) for e in tmp.replace('\n',' ').split()])#*constants.l_au2aa

    print('Attention: output is in au!')
    return symbols, coords, freqs, modes

