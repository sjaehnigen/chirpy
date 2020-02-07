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
import tempfile
import warnings
import sys
import copy

from ..physics import constants
from ..read.coordinates import cpmdIterator
from ..write.coordinates import xyzWriter

# SECTION > KEYWORD > ( <list of ARGUMENTS>, <next line (optional)> )
# PLANNED: multiple line support via list as next-line argument

# CONVERGENCE:
# should be float, but python does not undestand number format (e.g.,
# 2.0D-7); missing options: number of 2nd-line arguments


def cpmdReader(FN, **kwargs):
    '''Reads CPMD 4.3 files. Currently supported:
         GEOMETRY
         TRAJECTORY
         MOMENTS
      '''

    if 'filetype' not in kwargs:
        kwargs['filetype'] = FN
    filetype = kwargs.get('filetype')

    if any([_f in filetype for _f in [
                            'TRAJSAVED',
                            'GEOMETRY',
                            'TRAJECTORY',
                            'MOMENTS'
                            ]]):
        if ('symbols' in kwargs or 'numbers' in kwargs):
            numbers = kwargs.get('numbers')
            symbols = kwargs.get('symbols')
            if symbols is None:
                symbols = constants.numbers_to_symbols(numbers)
            kwargs.update({'symbols': symbols})
        else:
            raise TypeError("cpmdReader needs list of numbers or "
                            "symbols.")

        data = {}
        _load = np.array(tuple(cpmdIterator(FN, **kwargs)))
        data['data'] = _load
        data['comments'] = kwargs.get('comments', ['cpmd'] * _load.shape[0])
        data['symbols'] = symbols

        return data

    else:
        raise NotImplementedError('Unknown CPMD filetype %s!' % filetype)


def cpmdWriter(fn, data, append=False, **kwargs):
    '''Writes a CPMD TRAJECTORY or MOMENTS file including the frame column.
       Expects data of shape (n_frames, n_atoms, n_fields) in
       atomic units.

       Accepts frame=<int> or frames=<list> as optional frame info.

       write_atoms=True additionally writes an xyz file, containing data and
       symbols, as well as the ATOMS section for a CPMD input file
       (only append=False).
       '''

    bool_atoms = kwargs.get('write_atoms', True)

    if len(data.shape) == 2:
        # --- frame
        data = np.array([data])

    fmt = 'w'
    frame = kwargs.get('frame')
    frames = [frame]
    if frame is None:
        frames = kwargs.get('frames', range(data.shape[0]))
        if append:
            warnings.warn('Writing CPMD trajectory without frame info!',
                          stacklevel=2)

    if append:
        fmt = 'a'

    with open(fn, fmt) as f:
        for fr, _d in zip(frames, data):
            for _dd in _d:
                line = '%7d  ' % fr + '  '.join(map('{:22.14f}'.format, _dd))
                f.write(line+'\n')

    if bool_atoms and fmt != 'a':
        symbols = kwargs.pop('symbols', ())
        if data.shape[1] != len(symbols):
            raise ValueError('symbols and positions are not consistent!',
                             data.shape[1], symbols)
        if sorted(symbols) != list(symbols):
            raise ValueError('CPMD requires sorted data!')

        CPMDinput.ATOMS.from_data(symbols,
                                  data[0, :, :3],
                                  **kwargs).write_section(fn+'_ATOMS')
        xyzWriter(fn + '_ATOMS.xyz',
                  data[0, :, :3] / constants.l_aa2au,
                  symbols,
                  fn)


def cpmd_kinds_from_file(fn):
    '''Accepts MOMENTS or TRAJECTORY file and returns the number
       of lines per frame, based on analysis of the first frame'''

    warnings.warn('Automatic guess of CPMD kinds. Proceed with caution!',
                  stacklevel=2)
    with open(fn, 'r') as _f:
        _i = 1
        _fr = _f.readline().strip().split()[0]
        try:
            while _f.readline().strip().split()[0] == _fr:
                _i += 1
        except IndexError:
            pass

    return tuple(range(_i))


def fortran_float(a):
    '''Work in progress ...
       1.E-7 --> 1.0D-07'''
    if not isinstance(a, float):
        raise TypeError('Requires a float as input!')
    return


_cpmd_keyword_logic = {
    'INFO': {
        '': ([''], None),
    },
    'CPMD': {
        'CENTER MOLECULE': (['', 'OFF'], None),
        'CONVERGENCE ORBITALS': ([], str),
        'HAMILTONIAN CUTOFF': ([], float),
        'WANNIER PARAMETER': ([], str),  # 4 next-line arguments
        'RESTART': (['LATEST', 'WAVEFUNCTION', 'GEOFILE'], None),
        'LINEAR RESPONSE': ([], None),  # if set ask for RESP section
        'OPTIMIZE WAVEFUNCTION': ([], None),
        'MOLECULAR DYNAMICS': (['', 'FILE', 'NSKIP=-1'], None),
        'MIRROR': ([], None),
    },
    'RESP': {
        'CONVERGENCE': ([], str),
        'CG-ANALYTIC': ([], int),
        'NMR': (['NOVIRT', 'PSI0', 'CURRENT'], None),
        'MAGNETIC': (['VERBOSE'], None),
        'VOA': (['MD', 'CURRENT', 'ORBITALS'], None),
    },
    'DFT': {
        'NEWCODE': ([], None),
        'FUNCTIONAL': (['BLYP', 'PBE', 'PBE0'], None),
    },
    'SYSTEM': {
        'SYMMETRY': ([''], str),
        'CELL': (['', 'ABSOLUTE', 'DEGREE'], float),  # list of floats
        'CUTOFF': ([], float),
        'ANGSTROM': ([], None),
        'POISSON SOLVER': (['TUCKERMAN'], None),
    },
    'ATOMS': {
    },
}


class CPMDinput():
    class _SECTION():
        '''This is a universal class that parses any possible keyword input.
           Each derived class may contain a test routine that checks on
           superfluous or missing keywords
           ALPHA'''

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._dict = _cpmd_keyword_logic[self.__class__.__name__]
            self.options = {}
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
                    print("  "+" ".join([str(_a)
                                         for _a in self.options[_o][1]]))

            print("&END")

        def print_options(self):
            for _k in self._dict:
                print('%s: ' % _k, self._dict[_k])

        def set_keyword(self, keyword, *args, **kwargs):
            if keyword not in self._dict:
                raise AttributeError('Unknown keyword %s!' % keyword)
            _arglist, _nextline = self._dict[keyword]

            if any([_a not in _arglist for _a in args]):
                raise AttributeError(
                    'Unknown argument(s) for keyword %s: %s !'
                    % (keyword, [_a for _a in args if _a not in _arglist])
                    )

            _nl = None
            if _nextline is not None:
                subarg = kwargs.get('subarg')
                if not isinstance(subarg, list):
                    subarg = [subarg]
                if _nextline.__class__ is not list:
                    _nl = list(map(_nextline, subarg))
                else:
                    raise NotImplementedError(
                          'Multiple line keywords not supported: %s' % keyword)

            self.options.update({keyword: (args, _nl)})

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
                _key, _arg, _next = cls._parse_keyword_input(cls.__name__,
                                                             _line)
                _nl = None
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
        def set_keyword(self, name):
            '''No keywords in this section.'''
            self.set_name(name)

        def set_name(self, name):
            self.options = {str(name): ([], None)}

        @staticmethod
        def _parse_keyword_input(section, line):
            return line.strip(), [], None

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
                warnings.warn('Setting pseudopotential in CPMD ATOMS output to'
                              ' default (Troullier-Martins, BLYP)!',
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


class CPMDjob():
    '''An actual representative of CPMDinput'''

    def __init__(self, **kwargs):
        # mandatory
        self.INFO = CPMDinput.INFO(options={'CPMD DEFAULT JOB': ([], None)})
        for _sec in [
                'INFO',
                'CPMD',
                'SYSTEM',
                'DFT',
                'ATOMS',
                ]:
            setattr(self, _sec, kwargs.get(_sec, getattr(CPMDinput, _sec)()))

        # optional
        for _sec in [
                'RESP',
                # 'TRAJECTORY',
                ]:
            setattr(self, _sec, kwargs.get(_sec, getattr(CPMDinput, _sec)()))

        self._check_consistency()

    @classmethod
    def load_template(cls, template):
        '''Load the specified template and return a class object'''
        pass

    def _check_consistency(self):
        pass

    @classmethod
    def read_input_file(cls, fn):
        '''CPMD 4'''

        def _parse_file(_iter):
            _l = next(_iter)
            _c = []
            while "&END" not in _l:
                _c.append(_l)
                _l = next(_iter)
            return iter(_c)

        with open(fn, 'r') as _f:
            _iter = (_l.strip() for _l in _f)
            CONTENT = {_l[1:].upper(): _parse_file(_iter)
                       for _l in _iter if "&" in _l}

        CONTENT = {_C: getattr(CPMDinput, _C)._parse_section_input(CONTENT[_C])
                   for _C in CONTENT}

        return cls(**CONTENT)

    def write_input_file(self, *args, **kwargs):
        ''' CPMD 4 '''

        # known sections and order
        _SEC = ['INFO', 'CPMD', 'RESP', 'DFT', 'SYSTEM', 'ATOMS']

        for _s in _SEC:
            if hasattr(self, _s):
                getattr(self, _s).write_section(*args)

    # ToDo: into atoms

    def get_positions(self):
        ''' in a. u. ? '''
        return np.vstack(self.ATOMS.data)

    def get_symbols(self):
        symbols = []
        for _s, _n in zip(self.ATOMS.kinds, self.ATOMS.n_kinds):
            symbols += _s.split('_')[0][1:] * _n
        return np.array(symbols)

    def get_kinds(self):
        kinds = []
        for _s, _n in zip(self.ATOMS.kinds, self.ATOMS.n_kinds):
            kinds += [_s] * _n
        return np.array(kinds)

    def get_channels(self):
        channels = []
        for _s, _n in zip(self.ATOMS.channels, self.ATOMS.n_kinds):
            channels += [_s] * _n
        return np.array(channels)

    def split_atoms(self, ids):
        ''' Split atomic system into fragments and create CPMD job,
            respectively.
            ids ... list  of fragment id per atom
            (can be anything str, int, float, ...).
            '''
        def _dec(_P):
            return [[_P[_k] for _k, _jd in enumerate(ids) if _jd == _id]
                    for _id in sorted(set(ids))]

        # cpmd.TRAJECTORY is no longer in use !
        _L = []

        if hasattr(self, 'TRAJECTORY'):
            _getlist = [
                        self.get_positions(),
                        self.get_kinds(),
                        self.get_channels(),
                        self.TRAJECTORY.pos_aa.swapaxes(0, 1),
                        self.TRAJECTORY.vel_au.swapaxes(0, 1),
                      ]
        else:
            _getlist = [
                        self.get_positions(),
                        self.get_kinds(),
                        self.get_channels(),
                       ]

#        for _sd, _sk, _sch in zip(*map(_dec, _getlist)):
        for _frag in zip(*map(_dec, _getlist)):
            _C = {}
            _C['kinds'] = []
            _C['channels'] = []
            _C['n_kinds'] = []
            _C['data'] = []
            # complicated to keep order
            _isk_old = None
            # for _I, _isk in enumerate(sorted(set(_frag[1]))):
            for _isk in _frag[1]:
                if _isk != _isk_old:
                    _C['kinds'].append(_isk)
                    _C['channels'].append(
                            [_frag[2][_j]
                             for _j, _jsk in enumerate(_frag[1])
                             if _jsk == _isk
                             ][0])  # a little awkward
                    _C['n_kinds'].append(_frag[1].count(_isk))
                    _C['data'].append(
                            np.array([_frag[0][_j]
                                      for _j, _jsk in enumerate(_frag[1])
                                      if _jsk == _isk])
                            )
                _isk_old = copy.deepcopy(_isk)  # necessary?

            out = copy.deepcopy(self)
            out.ATOMS = CPMDinput.ATOMS(**_C)

            # if hasattr(self, 'TRAJECTORY'):
            #     out.TRAJECTORY = TRAJECTORY(
            #                         pos_aa=np.array(_frag[3]).swapaxes(0, 1),
            #                         vel_au=np.array(_frag[4]).swapaxes(0, 1),
            #                         symbols=out.get_symbols(),
            #                         )

            _L.append(out)

        return _L


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
