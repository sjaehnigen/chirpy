#!/usr/bin/env python3

import os
import sys
import numpy as np

#PDB Version 3.30 according to Protein Data Bank Contents Guide

def pdbReader(filename):
    print('WARNING BETA VERSION: Reading of occupancy and temp factor not yet implemented. I do not read the space group, either (i.e. giving P1)')
    names, resns,resids,data,symbols,cell_aa_deg,title = list(),list(),list(),list(), list(), None, None
    cell=0
    mk_int = lambda s: int(s) if s.strip() else 0

    for line in open(filename, 'r'):
        record = line[:6]

        if record == 'TITLE ':
            continuation  = line[ 8:10] # support for continuation to be added
            title = line[10:80].rstrip('\n')

        elif record == 'CRYST1':
            cell_aa_deg = np.array([e for e in map(float,[line[ 6:15],line[15:24],line[24:33],line[33:40],line[40:47],line[47:54]])])
            #data['space_group'] = line[55:66]
            #data['Z_value'    ] = int(line[66:70])
            cell=1

        elif record == 'ATOM  ' or record == 'HETATM':
            #atom_ser_nr.append(int(line[6:11]))
            names.append(line[12:16].strip())
            #alt_loc_ind.append(line[16])
            resns.append(line[17:20].strip())
            #Note: line[20] seems to be blank
            #chain_ind.append(line[21])
            resids.append(mk_int(line[22:26])) #residue sequence number
            #code_in_res.append(line[26])
            data.append([c for c in map(float,[line[30:38],line[38:46],line[46:54]])])
            #occupancy.append(float(line[54:60]))
            #temp_fact.append(float(line[60:66]))
            #seg_id.append(line[72:76])
            symbols.append(line[76:78].strip())
            #charges.append(line[78:80])

       # elif record == 'MASTER':
       #     numRemark = int(line[10:15])
       #     dummy     = int(line[15:20])
       #     numHet    = int(line[20:25])
       #     numHelix  = int(line[25:30])
       #     numSheet  = int(line[30:35])
       #     numTurn   = int(line[35:40])
       #     numSite   = int(line[40:45])
       #     numXform  = int(line[45:50])
       #     numCoord  = int(line[50:55])
       #     numTer    = int(line[55:60])
       #     numConect = int(line[60:65])
       #     numSeq    = int(line[65:70])
       # elif record.strip() == 'END':
       #     n_atoms = len(atom_ser_nr)

       #More records to come

    if cell==0:
        print('WARNING: No cell specified in pdb file!')
#    if box_aa_deg.all() == None:
#        raise Exception('Cell has to be specified, only orthorhombic cells supported!')
    return np.array(data), names, np.array(symbols), np.array([[resids[i],n] for i,n in enumerate(resns)]), cell_aa_deg, title

