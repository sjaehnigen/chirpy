#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


import numpy as np
import MDAnalysis as mda
# from CifFile import ReadCif
import warnings

from .generators import _reader


# --- kernels

def _xyz(frame, **kwargs):
    '''Kernel for processing xyz frame.'''

    _atomnumber = int(next(frame).strip())

    if kwargs.get('n_lines') != _atomnumber + 2:
        raise ValueError('Inconsistent XYZ file!')

    comment = next(frame).rstrip('\n')
    _split = (_l.strip().split() for _l in frame)
    symbols, data = zip(*[(_l[0], _l[1:]) for _l in _split])

    if len(data) != kwargs.get("n_lines") - 2:
        raise ValueError('Tried to read broken or incomplete file!')

    return np.array(data).astype(float), symbols, comment


def _cpmd(frame, **kwargs):
    '''Kernel for processing cpmd frame.'''

    filetype = kwargs.get('filetype')
    # --- is this a python bug?
    # --- generator needs at least one call of next() to work properly
    data = []
    data.append(next(frame).strip().split())

    for _l in frame:
        _l = _l.strip().split()
        data.append(_l)

    if len(data) != kwargs.get("n_lines"):
        raise ValueError('Tried to read broken or incomplete file!')

    if 'GEOMETRY' in filetype:
        return np.array(data).astype(float)

    elif any([f in filetype for f in ['TRAJSAVED', 'TRAJECTORY', 'MOMENTS']]):
        return np.array(data).astype(float)[:, 1:]

    else:
        raise ValueError('Unknown CPMD filetype %s' % filetype)

# --- This does currently not work (missing next() call)
#     if filetype == 'GEOMETRY':
#         return np.array([_l.strip().split() for _l in frame]).astype(float)
#
#     elif filetype in ['TRAJECTORY', 'MOMENTS']:
#         return np.array([_l.strip().split()[1:]
#                          for _l in frame]).astype(float)
#
#     else:
#         raise ValueError('Unknown CPMD filetype %s' % filetype)

# --- iterators


def xyzIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _xyz

    with open(FN, 'r') as _f:
        _nlines = int(_f.readline().strip()) + 2

    return _reader(FN, _nlines, _kernel, **kwargs)


def cpmdIterator(FN, **kwargs):
    '''Iterator for  cpmdReader
       Known types: GEOMETRY, TRAJECTORY, MOMENTS
       Usually expects additional metadata of the system
       through kwargs.'''
    _kernel = _cpmd
    symbols = kwargs.pop('symbols', None)
    if 'filetype' not in kwargs:
        kwargs['filetype'] = FN

    if symbols is None:
        _nlines = 1
    else:
        _nlines = len([_k for _k in symbols])  # type-independent

    return _reader(FN, _nlines, _kernel, **kwargs)

# --- complete readers


def xyzReader(FN, **kwargs):
    '''Read complete XYZ file at once.abs
       Returns data, symbols, comments of current frame'''
    data, symbols, comments = zip(*xyzIterator(FN, **kwargs))
    return np.array(data), symbols[0], list(comments)


# ------ external readers


def pdbIterator(FN):
    '''Iterator for pdbReader relying on MDAnalysis
       Usage: next() returns data, names, symbols, res,
       cell_aa_deg, title of current frame.

       https://www.mdanalysis.org/docs/documentation_pages/coordinates/PDB.html
       '''
    u = mda.Universe(FN)

    for ts in u.trajectory:
        # only take what is needed in ChirPy
        data = u.coord.positions
        resns = u.atoms.resnames
        resids = u.atoms.resids
        names = u.atoms.names
        symbols = tuple(u.atoms.types)
        cell_aa_deg = u.dimensions
        title = u.trajectory.title
        if np.prod(cell_aa_deg) == 0.0:
            warnings.warn('No or invalid cell specified in pdb file!',
                          RuntimeWarning)
            cell_aa_deg = None
        if len(title) == 0:
            title = None
        else:
            title = title[0]

        yield data, names, symbols, [_n for _n in zip(
                                             resids,
                                             resns)], cell_aa_deg, title


def pdbReader(FN, **kwargs):
    '''Read complete PDB file at once using MDAnalysis.
       Returns data, names, symbols, res, cell_aa_deg, title
       of current frame.

       https://www.mdanalysis.org/docs/documentation_pages/coordinates/PDB.html
       '''

    data, names, symbols, res, cell_aa_deg, title = \
        tuple([_b for _b in zip(*pdbIterator(FN, **kwargs))])

    return np.array(data), names[0], tuple(symbols[0]), tuple(res[0]), \
        cell_aa_deg[0], list(title)


# def cifReader(fn):
#    ReadCif(fn)
