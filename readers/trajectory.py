#!/usr/bin/env python

import numpy as np

#PDB Version 3.30 according to Protein Data Bank Contents Guide
#ToDo: file check routines for all Readers (e.g., "file is not a PDB file" )
#ToDo: integrity check for resulting data

def pdbReader(filename):
    '''WARNING BETA VERSION: Reading of occupancy and temp factor not yet implemented. I do not read the space group, either (i.e. giving P1)'''
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

##DEBUG 
def xyzReader(fn):
    """Adapted from Arne Scherrer's pythonbase ReadTrajectory_BruteForce(filename)
Input:  
        1. filename: File to read
Output: 
        1. np.array of shape (#frames, #atoms, #fields/atom)
        2. list of atom symbols (contains strings)
        3. list of comment lines (contains strings)"""
    f = open(fn, 'r')
    lines = f.readlines()
    f.close()

    n_atoms = int(lines[0].strip())
    n_frames = len(lines)//(n_atoms+2)
    data, symbols, comments = list(), list(), list()

    for n_frame in range(n_frames):
        offset = n_frame*(n_atoms + 2)

        # get comments
        comments.append(lines[offset+1].rstrip('\n'))

        # get symbols
        if n_frame == 0:
            symbols = [s.strip().split()[0] for s in lines[offset+2:offset+n_atoms+2]]

        # get data
        tmp = [[float(d) for d in s.strip().split()[1:]] for s in lines[offset+2:offset+n_atoms+2]]
        data.append(tmp)

    return np.array(data), symbols, comments

#not the final version: remove n_kinds dependency and generalise it
def cpmdReader(FN, kinds=[], type='GEOMETRY'):
    n_kinds = np.array(kinds).shape[0]
    with open(FN, 'r' ) as _f:
        _it = (list(map(float, line.strip().split())) for line in _f if 'NEW DATA' not in line)
        try:
            while _it:
                #pos, vel = tuple(np.array([next(_it) for _ik in range(n_kinds)]).reshape((n_kinds, 1, 6)).swapaxes(0, 1))
                #yield np.array( [ next( _it ) for _ik in range( n_kinds ) ] ).reshape( ( n_kinds, 1, 3 ) ).swapaxes( 0, 1 ) 
                yield tuple(np.array([next(_it) for _ik in range(n_kinds)]).reshape((n_kinds, 6)))
                        #.reshape((n_kinds, 1, 6)))
        except StopIteration:
            pass



class cpmdReader1():
    """iterates over FN and yields generator of positions, velocities and moments (in a.u.)"""
    def __init__(self, *args, **kwargs):
        if len( args ) != 2:
            raise TypeError('cpmdReader takes two arguments: filename and kinds')
        self.type = kwargs.get('type', 'GEOMETRY')
        self.fn = args[0]
        self.kinds = np.array(args[1])
        self.n_kinds = self.kinds.shape[0]

#    def _gen(self):
        with open(self.fn, 'r' ) as _f:
            self.gen = (list(map(float, line.strip().split())) for line in _f if 'NEW DATA' not in line)
#            try:
#                while _it:
#                    yield next(_it)
#            except StopIteration:
#                pass

    def __iter__(self):
        return self

    def __next__(self):
#        with open(self.fn, 'r' ) as _f:
#            _gen = (list(map(float, line.strip().split())) for line in _f if 'NEW DATA' not in line)
            if self.type == 'GEOMETRY':
                try:
                    while self.gen:
                        yield tuple(np.array([next(self.gen) for _ik in range(self.n_kinds)]))
                except StopIteration:
                    pass

            if self.type == 'TRAJECTORY':
                #while _gen:
                return tuple(np.array([next(self._gen())[1:] for _ik in range(self.n_kinds)]).reshape((self.n_kinds, 1, 6)))

            raise StopIteration
            #try:
            #    while self._gen:
            #        pos, vel = tuple(np.array([next(_it) for _ik in range(n_kinds)]).reshape((n_kinds, 2, 3)).swapaxes(0, 1))
            #        #yield np.array( [ next( _it ) for _ik in range( n_kinds ) ] ).reshape( ( n_kinds, 1, 3 ) ).swapaxes( 0, 1 ) 
            #        yield pos, vel
            #except StopIteration:
            #    pass

