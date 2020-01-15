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
import tempfile
import warnings
import sys

# from ..write.coordinates import cpmdWriter
# from ..read.coordinates import cpmdIterator
from ..physics import constants
from ..read.coordinates import cpmdIterator
from ..write.coordinates import xyzWriter

# SECTION > KEYWORD > ( <list of ARGUMENTS>, <next line (optional)> )
# PLANNED: multiple line support via list as next-line argument

# CONVERGENCE:
# should be float, but python does not undestand number format (e.g.,
# 2.0D-7); missing options: number of 2nd-line arguments

# NB: CPMD writes either pos_aa + vel_aa or *_au, regardless the file format
# (xyz convention here is pos_aa/vel_au though...)

_cpmd_keyword_logic = {
    'INFO': {
    },
    'CPMD': {
        'CENTER MOLECULE': (['', 'OFF'], None),
        'CONVERGENCE ORBITALS': ([''], str),
        'HAMILTONIAN CUTOFF': ([''], float),
        'WANNIER PARAMETER': ([''], str),  # 4 next-line arguments
        'RESTART': (['LATEST', 'WAVEFUNCTION', 'GEOFILE'], None),
        'LINEAR RESPONSE': ([''], None),
        'OPTIMIZE WAVEFUNCTION': ([''], None),
    },
    'RESP': {
        'CONVERGENCE': ([''], str),
        'CG-ANALYTIC': ([''], int),
        'NMR': (['NOVIRT', 'PSI0', 'CURRENT'], None),
        'MAGNETIC': (['VERBOSE'], None),
        'VOA': (['MD', 'CURRENT', 'ORBITALS'], None),
    },
    'DFT': {
        'NEWCODE': ([''], None),
        'FUNCTIONAL': (['BLYP', 'PBE', 'PBE0'], None),
    },
    'SYSTEM': {
        'SYMMETRY': ([''], str),
        'CELL': (['', 'ABSOLUTE'], float),  # list of floats
        'CUTOFF': ([''], float),
        'ANGSTROM': ([''], None),
    },
    'ATOMS': {
    },
}


class _SECTION():
    '''This is a universal class that parses all possible keywords. Each derived
       class may contain a test routine that checks on superfluous or missing
       keywords'''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
#        for _i in kwargs:
#            setattr(self, _i, kwargs[_i])

    def write_section(self, *args, **kwargs):
        _stdout = sys.stdout
        # _sys.__stdout__ is not Jupyter Output,
        # so use this way to restore stdout

        if len(args) == 1:
            sys.stdout = open(args[0], 'a')
        elif len(args) > 1:
            raise TypeError(self.write_section.__name__ +
                            ' takes at most 1 argument.')

        self.print_section(**kwargs)

        sys.stdout.flush()
        # _sys.stdout = _sys.__stdout__
        sys.stdout = _stdout

    def print_section(self, **kwargs):
        print("&%s" % self.__class__.__name__)

        for _o in self.options:
            print(" "+" ".join([_o] + [_a for _a in self.options[_o][0]]))

            # print next-line argument
            if self.options[_o][1] is not None:
                print("  "+" ".join([str(_a) for _a in self.options[_o][1]]))

        print("&END")

    @staticmethod
    def _parse_keyword_input(section, line):
        _section_keys = _cpmd_keyword_logic[section]

        _key = [_k for _k in _section_keys if _k in line[:len(_k)]]

        if len(_key) == 0:
            raise AttributeError('Unknown keyword %s!' % _key)
        elif len(_key) > 1:
            print('WARNING: Found multiple keywords in line %s' % _key)
        else:
            _key = _key[0]
            _arg = line.split(_key)[1].strip().split()
            _arglist, _nextline = _section_keys[_key]

            if any(_a not in _arglist for _a in _arg):
                raise Exception(
                        'Unknown arguments for keyword %s: %s !'
                        % (_key, [_a for _a in _arg if _a not in _arglist])
                        )

        return _key, _arg, _nextline

    @classmethod
    def _parse_section_input(cls, section_input):
        '''Default minimal parser.
           Takes unprocessed lines from section without &<SECTION> and
           "&END" lines as section_input.'''
        _C = {}

        options = {}
        for _line in section_input:
            _key, _arg, _next = cls._parse_keyword_input(cls.__name__, _line)
            if _next is not None:
                if _next.__class__ is not list:
                    _nl = list(map(_next, next(section_input).split()))
                else:
                    raise NotImplementedError(
                            'Multiple line keywords not supported: %s' % _key)
                options.update({_key: (_arg, _nl)})
        _C['options'] = options

        out = cls()
        out.__dict__.update(_C)
        return out


class INFO(_SECTION):
    pass


class CPMD(_SECTION):
    pass


class RESP(_SECTION):
    pass


class DFT(_SECTION):
    pass


class SYSTEM(_SECTION):
    pass


class ATOMS(_SECTION):
    def print_section(self, **kwargs):
        fmt = kwargs.get('fmt', 'angstrom')
        print("&%s" % self.__class__.__name__)

        format = '%20.10f' * 3
        for _k, _ch, _n, _d in zip(self.kinds,
                                   self.channels,
                                   self.n_kinds,
                                   self.data):
            print("%s" % _k)
            print(" %s" % _ch)
            print("%6d" % _n)
            for _dd in _d:
                if fmt == 'angstrom':
                    warnings.warn('Atomic coordinates in angstrom. Do not '
                                  'forget to set ANGSTROM keyword in the '
                                  'SYSTEM section!', stacklevel=2)
                    _dd *= constants.l_au2aa
                print(format % tuple(_dd))

        print("&END")

    @classmethod
    def _parse_section_input(cls, section_input):
        '''Takes unprocessed lines from section without &<SECTION> and
           "&END" lines as section_input.
           Beta: Limited LOC/LMAX support.'''
        _C = {}
        kinds = []
        channels = []
        n_kinds = []
        data = []
        for _l in section_input:
            if '*' in _l:
                kinds.append(_l.strip())
                channels.append(next(section_input).strip())
                n = int(next(section_input))
                n_kinds.append(n)
                data.append(np.array([list(map(
                                        float,
                                        next(section_input).split()
                                        )) for _i in range(n)]))
        _C['kinds'] = kinds
        _C['channels'] = channels
        _C['n_kinds'] = n_kinds
        _C['data'] = data

        out = cls()
        out.__dict__.update(_C)
        return out

    @classmethod
    def from_data(cls, symbols, pos_au, **kwargs):
        if not hasattr(kwargs, 'pp'):
            warnings.warn('Setting pseudopotential in CPMD ATOMS output to '
                          'default (Troullier-Martins)!',
                          stacklevel=2)
        pp = kwargs.get('pp', 'MT_BLYP')  # 'SG_BLYP KLEINMAN-BYLANDER'

        elements = sorted(set(symbols))
        symbols = np.array(symbols)

        kinds = []
        n_kinds = []
        channels = []
        data = []
        for _e in elements:
            kinds.append("*%s_%s" % (_e, pp))
            data.append(pos_au[symbols == _e])
            n_kinds.append(len(data[-1]))
            if _e in ['C', 'O', 'N', 'P', 'Cl', 'F'] and 'AEC' not in pp:
                # --- ToDo: replace manual tweaks by automatic pp analysis
                channels.append("LMAX=P LOC=P")
            else:
                channels.append("LMAX=S LOC=S")

        _C = {}
        _C['kinds'] = kinds
        _C['channels'] = channels
        _C['n_kinds'] = n_kinds
        _C['data'] = data

        out = cls()
        out.__dict__.update(_C)
        return out


#def _write_atoms_section(fn, symbols, pos_au, **kwargs):
#    '''Only sorted data is comaptible with CPMD.'''
#    if not hasattr(kwargs, 'pp'):
#        warnings.warn('Setting pseudopotential in CPMD ATOMS output to default'
#                      ' (Troullier-Martins)!',
#                      stacklevel=2)
#    pp = kwargs.get('pp', 'MT_BLYP')  # 'SG_BLYP KLEINMAN-BYLANDER'
#    fmt = kwargs.get('fmt', 'angstrom')
#    elems = dict()
#    if pos_au.shape[0] != len(symbols):
#        raise ValueError('symbols and positions are not consistent!')
#
#    pos = copy.deepcopy(pos_au)
#    if fmt == 'angstrom':
#        pos /= constants.l_aa2au
#
#    for i, sym in enumerate(symbols):
#        if sym != sorted(symbols)[i]:
#            raise ValueError('Atom list not sorted!')
#        try:
#            elems[sym]['n'] += 1
#            elems[sym]['c'][elems[sym]['n']] = pos[i]
#        except KeyError:
#            elems[sym] = dict()
#            elems[sym]['n'] = 1
#            elems[sym]['c'] = {elems[sym]['n']: pos[i]}
#
#    with open(fn, 'w') as f:
#        format = '%20.10f'*3 + '\n'
#        f.write("&ATOMS\n")
##        f.write(" ISOTOPE\n")
##        for elem in elems:
##            print("  %s" % elems[elem]['MASS'])
#        for elem in elems:
#            f.write("*%s_%s\n" % (elem, pp))
#            # --- ToDo: replace manual tweaks by automatic pp analysis
#            if elem in ['C', 'O', 'N', 'P', 'Cl', 'F'] and 'AEC' not in pp:
#                f.write(" LMAX=P LOC=P\n")
#            else:
#                f.write(" LMAX=S LOC=S\n")
#            f.write("   %s\n" % elems[elem]['n'])
#            for i in elems[elem]['c']:
#                f.write(format % tuple([c for c in elems[elem]['c'][i]]))
#        f.write("&END\n")


def cpmdReader(FN, **kwargs):
    '''Reads CPMD 4.3 files. Currently supported:
         GEOMETRY
         TRAJECTORY
         MOMENTS
      '''

    if 'filetype' not in kwargs:
        kwargs['filetype'] = FN
    filetype = kwargs.get('filetype')

    if filetype in ['GEOMETRY', 'TRAJECTORY', 'MOMENTS']:
        if 'kinds' not in kwargs:
            if ('symbols' in kwargs or 'numbers' in kwargs):
                numbers = kwargs.get('numbers')
                symbols = kwargs.get('symbols')
                if symbols is None:
                    symbols = constants.numbers_to_symbols(numbers)
                kwargs.update({'kinds': symbols})
            else:
                raise TypeError("cpmdReader needs list of kinds, numbers or "
                                "symbols.")

        data = {}
        _load = np.array(tuple(cpmdIterator(FN, **kwargs)))
        data['data'] = _load
        data['symbols'] = kwargs.get('kinds')
        data['comments'] = kwargs.get('comments', ['cpmd'] * _load.shape[0])

        return data

    else:
        raise NotImplementedError('Unknown CPMD filetype %s!' % filetype)


def cpmdWriter(fn, pos_au, vel_au, append=False, **kwargs):
    '''Expects pos_au / vel_au of shape (n_frames, n_atoms, three)'''

    bool_atoms = kwargs.get('write_atoms', True)

    fmt = 'w'
    if append:
        fmt = 'a'
    with open(fn, fmt) as f:
        format = ' %16.12f'*3
        for fr in range(pos_au.shape[0]):
            for at in range(pos_au.shape[1]):
                line = '%06d  ' % fr + format % tuple(pos_au[fr, at])
                line += '  ' + format % tuple(vel_au[fr, at])
                f.write(line+'\n')

    if bool_atoms:
        symbols = kwargs.pop('symbols', ())
        if pos_au.shape[1] != len(symbols):
            raise ValueError('symbols and positions are not consistent!')

        ATOMS.from_data(symbols,
                        pos_au[0],
                        **kwargs).write_section(fn+'_ATOMS')
        xyzWriter(fn + '_ATOMS.xyz',
                  [pos_au[0] / constants.l_aa2au],
                  symbols,
                  [fn])


def cpmd_kinds_from_file(fn):
    '''Accepts MOMENTS or TRAJECTORY file and returns the number
       of lines per frame, based on analysis of the first frame'''

    warnings.warn('Automatic guess of CPMD kinds. Proceed with caution!',
                  stacklevel=2)
    with open(fn, 'r') as _f:
        _i = 1
        _fr = _f.readline().strip().split()[0]
        while _f.readline().strip().split()[0] == _fr:
            _i += 1

    return tuple(range(_i))

#    if pos_au.shape[0] != len(symbols):
#        print('ERROR: symbols and positions are not consistent!')
#        sys.exit(1)
#
#    pos = copy.deepcopy(pos_au) #copy really necessary?
#    if fmt=='angstrom': pos /= Angstrom2Bohr
#
#    for i,sym in enumerate(symbols):
#        if sym != sorted(symbols)[i]:
#            print('ERROR: Atom list not sorted!')
#            sys.exit(1)
#        try:
#            elems[sym]['n'] +=1
#            elems[sym]['c'][elems[sym]['n']] = pos[i]
#        except KeyError:
#            if sym in constants.symbols:
#                elems[sym] = constants.species[sym]
#            elems[sym] = OrderedDict()
#            elems[sym]['n'] = 1
#            elems[sym]['c'] = {elems[sym]['n'] : pos[i]}
#            else: raise Exception("Element %s not found!" % sym)


# class TRAJECTORY(_TRAJ):
#     # --- DEPRECATED ?
#     '''Convention: pos in a.a., vel in a.u.
#        Use keyword to switch between representations '''
#
#     def __init__(self, **kwargs):
#         for _i in kwargs:
#             setattr(self, _i, kwargs[_i])
#         self._sync_class()
#
#     @classmethod
#     def read(cls, fn, n_atoms, **kwargs):
#         '''Beta: needs symbols'''
#         data = np.array([_d for _d in cpmdIterator(
#                                         fn,
#                                         n_atoms,
#                                         kwargs.get('type', 'TRAJECTORY')
#                                         )])
#         pos, vel = tuple(data.swapaxes(0, 1))
#         return cls(
#                 pos_aa=pos*constants.l_au2aa,
#                 vel_au=vel,
#                 symbols=kwargs.get('symbols'),
#                 )

#    def write(self, fn, **kwargs): #Later: Use global write function of _TRAJ
#        sym = ['X'] * self.pos.shape[1]
#        # ToDo: updat..writers: symbols not needed for TRAJSAVED output
#        cpmdWriter(fn, self.pos, sym, self.vel, write_atoms = False)


def extract_mtm_data_tmp(MTM_DATA_E0, MTM_DATA_R1, n_frames, n_states):
    '''Temporary version for debugging MTM. Demands CPMD3 output file.'''
    fn_buf1 = tempfile.TemporaryFile(dir='/tmp/')
    fn_buf2 = tempfile.TemporaryFile(dir='/tmp/')

    buf1 = np.memmap(fn_buf1,
                     dtype='float64',
                     mode='w+',
                     shape=(n_frames*n_states*n_states))

    with open(MTM_DATA_E0, 'r') as f:
        for i, line in enumerate(f):
            buf1[i] = float(line.strip().split()[-1])

    buf2 = np.memmap(fn_buf2,
                     dtype='float64',
                     mode='w+',
                     shape=(n_frames*n_states*n_states, 3))

    with open(MTM_DATA_R1, 'r') as f:
        for i, line in enumerate(f):
            buf2[i] = np.array(line.strip().split()[-3:]).astype(float)

    E0 = buf1.reshape((n_frames, n_states, n_states))
    R1 = buf2.reshape((n_frames, n_states, n_states, 3))

    del buf1, buf2

# Factor 2 already in CPMD --> change routine later
    return E0/2, R1/2
