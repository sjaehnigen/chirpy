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

from ..classes.trajectory import _TRAJECTORY as _TRAJ
# from ..writers.coordinates import cpmdWriter
from ..readers.coordinates import cpmdReader
from ..physics import constants

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

    def print_section(self):
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
    def print_section(self):
        print("&%s" % self.__class__.__name__)

        format = '%20.10f'*3
        for _k, _ch, _n, _d in zip(self.kinds,
                                   self.channels,
                                   self.n_kinds,
                                   self.data):
            print("%s" % _k)
            print(" %s" % _ch)
            print("%4d" % _n)
            for _dd in _d:
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
                kinds.append(_l)
                channels.append(next(section_input))
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


class TRAJECTORY(_TRAJ):
    '''Convention: pos in a.a., vel in a.u.
       Use keyword to switch between representations '''

    def __init__(self, **kwargs):
        for _i in kwargs:
            setattr(self, _i, kwargs[_i])
        self._sync_class()

    @classmethod
    def read(cls, fn, n_atoms, **kwargs):
        '''Beta: needs symbols'''
        data = np.array([_d for _d in cpmdReader(
                                        fn,
                                        n_atoms,
                                        kwargs.get('type', 'TRAJECTORY')
                                        )])
        pos, vel = tuple(data.swapaxes(0, 1))
        return cls(
                pos_aa=pos*constants.l_au2aa,
                vel_au=vel,
                symbols=kwargs.get('symbols'),
                )

#    def write(self, fn, **kwargs): #Later: Use global write function of _TRAJ
#        sym = ['X'] * self.pos.shape[1]
#        # ToDo: updat..writers: symbols not needed for TRAJSAVED output
#        cpmdWriter(fn, self.pos, sym, self.vel, write_atoms = False)


def get_frame_traj_and_mom(TRAJ, MOMS, n_atoms, n_moms):
    """Iterates over TRAJECTORY and MOMENTS files and
    yields generator of positions, velocities and moments (in a.u.)"""

    with open(TRAJ, 'r') as traj_f, open(MOMS, 'r') as moms_f:
        traj_it = (list(map(float, line.strip().split()[1:]))
                   for line in traj_f if 'NEW DATA' not in line)
        moms_it = (list(map(float, line.strip().split()[1:]))
                   for line in moms_f if 'NEW DATA' not in line)
        try:
            while traj_it and moms_it:
                pos_au, vel_au = tuple(np.array(
                    [next(traj_it) for i_atom in range(n_atoms)]
                    ).reshape((n_atoms, 2, 3)).swapaxes(0, 1))
                wc_au, c_au, m_au = tuple(np.array(
                    [next(moms_it) for i_mom in range(n_moms)]
                    ).reshape((n_moms, 3, 3)).swapaxes(0, 1))

                yield pos_au, vel_au, wc_au, c_au, m_au

        except StopIteration:
            pass


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
