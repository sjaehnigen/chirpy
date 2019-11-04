#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


import numpy as np


def _gen(fn):
    '''Global generator for all formats'''
    return (line for line in fn if 'NEW DATA' not in line)


def _get(_it, kernel, **kwargs):
    n_lines = kwargs.get('n_lines')
    r0, r1 = kwargs.pop("range", (0, float('inf')))
    _r = 0

    while _r < r0:
        [next(_it) for _ik in range(n_lines)]
        _r += 1

    while True:
        try:
            yield kernel([next(_it) for _ik in range(n_lines)], **kwargs)
            _r += 1
            if _r >= r1:
                raise StopIteration()
        except StopIteration:
            break


def _reader(FN, _nlines, _kernel, **kwargs):
    kwargs.update({'n_lines': _nlines})
    with open(FN, 'r') as _f:
        _it = _gen(_f)
        data = tuple(_get(_it, _kernel, **kwargs))
        if np.size(data) == 0:
            raise ValueError('Given input and arguments \
                    do not yield any data!')
        else:
            return data


def _xyz(frame, **kwargs):
    if kwargs.get('n_lines') != int(frame[0].strip()) + 2:
        raise ValueError('Corrupt XYZ file!')

    comment = frame[1].rstrip('\n')
    _split = (_l.strip().split() for _l in frame[2:])
    symbols, data = tuple(zip(*[(_l[0], _l[1:]) for _l in _split]))

    return np.array(data).astype(float), symbols, comment


def _cpmd(frame, **kwargs):
    """Iterates over FN and yields generator of positions, \
           velocities and/or moments (in a.u.)"""
    filetype = kwargs.get('filetype')
    if filetype == 'GEOMETRY':
        return np.array([_l.strip().split() for _l in frame]).astype(float)
    elif filetype in ['TRAJECTORY', 'MOMENTS']:
        return np.array([_l.strip().split()[1:] for _l in frame]).astype(float)
    else:
        raise TypeError('Unknown filetype %s' % filetype)


def xyzReader(FN, **kwargs):
    _kernel = _xyz

    with open(FN, 'r') as _f:
        _nlines = int(_f.readline().strip()) + 2

    data, symbols, comments = zip(*_reader(FN, _nlines, _kernel, **kwargs))
    return np.array(data), symbols[0], list(comments)


def cpmdReader(FN, **kwargs):
    _kernel = _cpmd

    _nlines = np.array(kwargs.get('kinds', [0])).shape[0]

    return _reader(FN, _nlines, _kernel, **kwargs)


# ------ old readers

def pdbReader(filename):
    '''PDB Version 3.30 according to Protein Data Bank Contents Guide.
    WARNING BETA VERSION: Reading of occupancy and temp factor not yet implemented. I do not read the space group, either (i.e. giving P1)'''
    names, resns,resids,data,symbols,cell_aa_deg,title = list(),list(),list(),list(), list(), None, None
    cell=0
    mk_int = lambda s: int(s) if s.strip() else 0

    #define error logs
    _e0 = 0 #file integrity

    for line in open(filename, 'r'):
        record = line[:6]

        if record == 'TITLE ':
            #continuation  = line[ 8:10] # support for continuation to be added
            title = line[10:80].rstrip('\n')

        elif record == 'CRYST1':
            cell_aa_deg = np.array([e for e in map(float,[line[ 6:15],line[15:24],line[24:33],line[33:40],line[40:47],line[47:54]])])
            #data['space_group'] = line[55:66]
            #data['Z_value'    ] = int(line[66:70])
            cell=1

        elif record == 'ATOM  ' or record == 'HETATM':
            #atom_ser_nr.append(int(line[6:11]))
            names.append( line[ 12 : 16 ].strip() )
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
            _s = line[ 76 : 78 ].strip()
            if _s == '':
                _e0 += 1
                symbols.append( names[ -1 ][ :2 ] )
            else:
                symbols.append( line[ 76 : 78 ].strip() )
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
    #evaluate error logs
    if _e0 != 0:
        print( '\nWARNING: Incomplete or corrupt PDB file. Proceed with care!\n' )

    if cell==0:
        print('WARNING: No cell specified in pdb file!')
#    if box_aa_deg.all() == None:
#        raise Exception('Cell has to be specified, only orthorhombic cells supported!')
    return np.array(data), names, np.array(symbols), np.array([[i,n] for i,n in zip(resids,resns)]), cell_aa_deg, title

