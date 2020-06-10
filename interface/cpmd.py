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

import os
import numpy as np
import tempfile
import warnings
import sys
import copy

from ..physics import constants
from ..read.coordinates import cpmdIterator
from ..read.generators import _open
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
         MOLVIB
         APT
         AAT
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

    elif 'MOLVIB' in filetype:
        # ToDo: OLD CODE (not tested)
        with open(FN, 'r') as f:
            inbuffer = f.read()
        inbuffer.split()
        sep = inbuffer.index('&END')+5
        cart_block = inbuffer[:sep].split()[1:-1]
        hessian_block = inbuffer[sep:].split()[1:-1]
        cart_block = np.array([float(e) for e in cart_block])
        hessian_block = np.array([float(e) for e in hessian_block])
        n_atoms = len(cart_block) // 5
        cart_block = cart_block.reshape((n_atoms, 5)).transpose()
        numbers = cart_block[0]
        coords = cart_block[1:-1].transpose()*constants.l_au2aa
        masses = cart_block[-1]
        hessian = hessian_block.reshape((n_atoms*3, n_atoms*3))

        data = {}
        data['symbols'] = constants.numbers_to_symbols(numbers)
        data['data'] = coords
        data['masses'] = masses
        data['hessian'] = hessian
        # modes ?

    elif 'APT' in filetype:
        # ToDo: OLD CODE (not tested)
        data = {}
        dat = np.genfromtxt(FN, dtype=None, autostrip=True)
        data['AAT'] = dat.reshape(dat.shape[0] // 3, 9)

    elif 'AAT' in filetype:
        # ToDo: OLD CODE (not tested)
        data = {}
        dat = np.genfromtxt(FN, dtype=None, autostrip=True)
        data['AAT'] = dat.reshape(dat.shape[0] // 3, 9)

    # elif 'POLARIZATION' in filetype:
    #     with open(FN, 'r') as f:
    #         for line in f.readlines():
    #             if 'atomic units' in line:
    #                 return np.array([float(e) for e in line.split()[:3]])
    #     raise Exception('No valid output found! %s'%fn_property_job)

    else:
        raise NotImplementedError('Unknown CPMD filetype %s!' % filetype)

    return data


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

    mode = 'w'
    frame = kwargs.get('frame')
    frames = [frame]
    if frame is None:
        frames = kwargs.get('frames', range(data.shape[0]))
        if append:
            warnings.warn('Writing CPMD trajectory without frame info!',
                          stacklevel=2)

    if append:
        mode = 'a'

    with open(fn, mode) as f:
        for fr, _d in zip(frames, data):
            for _dd in _d:
                line = '%7d  ' % fr + '  '.join(map('{:22.14f}'.format, _dd))
                f.write(line+'\n')

    if bool_atoms and mode != 'a':
        symbols = kwargs.pop('symbols', ())
        if data.shape[1] != len(symbols):
            raise ValueError('symbols and positions are not consistent!',
                             data.shape[1], symbols)
        if sorted(symbols) != list(symbols):
            raise ValueError('CPMD requires sorted data!')

        CPMDinput.ATOMS.from_data(symbols,
                                  data[0, :, :3],
                                  pp=kwargs.get('pp'),
                                  ).write_section(fn+'_ATOMS')
        xyzWriter(fn + '_ATOMS.xyz',
                  data[0, :, :3] / constants.l_aa2au,
                  symbols,
                  fn)


# ToDo: OLD CODE
def write_molvib(filename, n_atoms, numbers, coordinates, masses, hessian):
    obuffer = '&CART\n'
    for n, r, m in zip(numbers, coordinates, masses):
        obuffer += ' %d  %16.12f  %16.12f  %16.12f  %16.12f\n' % tuple(
                                          [n]+list(r*constants.l_aa2au)+[m])
    obuffer += '  &END \n&FCON \n'
    hessian = hessian.reshape((n_atoms*n_atoms*3, 3))
    for line in hessian:
        obuffer += ' %16.12f  %16.12f  %16.12f\n' % tuple(line)
    obuffer += '&END \n'
    with open(filename, 'w') as f:
        f.write(obuffer)


def write_atomic_tensor(at_filename, atomic_tensor):
    (dim1, dim2) = atomic_tensor.shape
    if dim2 != 9:
        raise Exception('Invalid format of at!')
    atomic_tensor = atomic_tensor.reshape(dim1, 3, 3)
    with open(at_filename, 'w') as f:
        for i in range(dim1):
            for j in range(3):
                f.write(' %20.15f %20.15f %20.15f\n' % tuple(
                                             atomic_tensor[i, j]))


def cpmd_kinds_from_file(fn, **kwargs):
    '''Accepts MOMENTS or TRAJECTORY file and returns the number
       of lines per frame, based on analysis of the first frame'''

    warnings.warn('Automatic guess of CPMD kinds. Proceed with caution!',
                  stacklevel=2)
    with _open(fn, 'r') as _f:
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


def _nextline_parser(SEC, KEY, ARG, section_input):
    def _fmt(s):
        return '%10.5f' % float(s)
    if SEC == 'SYSTEM':
        if KEY == 'CELL':
            if 'VECTORS' in ARG:
                _out = [tuple(map(_fmt, _s.split()))
                        for _iv, _s in zip(range(3), section_input)]
            else:
                _out = tuple(map(_fmt, next(section_input).split()))
            return _out, section_input


_cpmd_keyword_logic = {
    'INFO': {
        '': ([''], None),
    },
    'CPMD': {
        'CENTER MOLECULE': (['', 'OFF'], None),
        'CONVERGENCE': (['ORBITALS', 'GEOMETRY', 'CELL'], str),
        'HAMILTONIAN CUTOFF': ([], float),
        'WANNIER PARAMETER': ([], str),  # 4 next-line arguments
        'RESTART': (['LATEST', 'WAVEFUNCTION', 'GEOFILE', 'DENSITY', 'ALL',
                     'COORDINATES'],
                    None),
        'LINEAR RESPONSE': ([], None),  # if set ask for RESP section
        'OPTIMIZE WAVEFUNCTION': ([], None),
        'OPTIMIZE GEOMETRY': (['XYZ', 'SAMPLE'], None),
        'MOLECULAR DYNAMICS': (
            ['BO', 'CP', 'EH', 'PT', 'CLASSICAL', 'FILE', 'XYZ', 'NSKIP=-1'],
            None
            ),
        'MIRROR': ([], None),
        'RHOOUT': (['BANDS', 'SAMPLE'], [int, float]),
        'TIMESTEP': ([], float),
    },
    'RESP': {
        'CONVERGENCE': ([], str),
        'CG-ANALYTIC': ([], int),
        'CG-FACTOR': ([], float),
        'NMR': (['NOVIRT', 'PSI0', 'CURRENT'], None),
        'MAGNETIC': (['VERBOSE'], None),
        'VOA': (['MD', 'CURRENT', 'ORBITALS', 'DENSITY', 'HALFMESH'], None),
        'EPR': ([], [str, str]),
        'POLAK': ([], None),
    },
    'DFT': {
        'NEWCODE': ([], None),
        'FUNCTIONAL': (['NONE', 'LDA', 'PADE', 'GGA', 'PW91', 'BP', 'BLYP',
                        'B1LYP', 'B3LYP', 'HSE06', 'OLYP', 'PBE', 'PBES',
                        'PBE0', 'REVPBE'], None),
    },
    'SYSTEM': {
        'SYMMETRY': ([''], str),
        'CELL': (['', 'ABSOLUTE', 'DEGREE', 'VECTORS'], _nextline_parser),
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
            fmt = kwargs.get('fmt', 'angstrom')
            append = kwargs.get('append', False)

            if len(args) == 1:
                mode = 'w'
                if append:
                    mode = 'a'
                _outstream = open(args[0], mode)

            if len(args) > 1:
                raise TypeError(self.write_section.__name__ +
                                ' takes at most 1 argument.')

            self.print_section(file=_outstream, fmt=fmt)

        def print_section(self, file=sys.stdout, **kwargs):
            print("&%s" % self.__class__.__name__, file=file)

            for _o in self.options:
                print(" "+" ".join([_o] + [_a for _a in self.options[_o][0]]),
                      file=file)

                # --- print next-line argument
                if self.options[_o][1] is not None:
                    _nl = self.options[_o][1]
                    if not isinstance(_nl, list):
                        _nl = [_nl]
                    for _l in _nl:
                        print("  "+" ".join([str(_a) for _a in _l]), file=file)

            print("&END", file=file)

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
                raise AttributeError('Unknown keyword for section %s: %s!'
                                     % (section, line))
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
        def _parse_section_input(cls, section_input, fmt='angstrom'):
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
                    if _next.__name__ in ('str', 'float', 'int'):
                        # ToDo: put a function here
                        _nl = tuple(map(_next, next(section_input).split()))
                    elif callable(_next):
                        _nl, section_input = _next(cls.__name__, _key, _arg,
                                                   section_input)
                    else:
                        raise TypeError(
                             'CPMD input lines not understood for %s' % _key)
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
        def print_section(self, fmt='angstrom', file=sys.stdout):
            print("&%s" % self.__class__.__name__, file=file)

            format = '%20.10f' * 3
            for _k, _ch, _n, _d in zip(self.kinds,
                                       self.channels,
                                       self.n_kinds,
                                       self.data):
                print("%s" % _k, file=file)
                print(" %s" % _ch, file=file)
                print("%6d" % _n, file=file)
                for _dd in _d:
                    if fmt == 'angstrom':
                        warnings.warn('Atomic coordinates in angstrom. Do not '
                                      'forget to set ANGSTROM keyword in the '
                                      'SYSTEM section!', stacklevel=2)
                        _dd *= constants.l_au2aa
                    print(format % tuple(_dd), file=file)

            print("&END", file=file)

        @classmethod
        def _parse_section_input(cls, section_input, fmt='angstrom'):
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

            # --- simple solution for pp
            _C['pp'] = '_'.join(kinds[0].split('_')[1:])
            _C['kinds'] = kinds
            _C['channels'] = channels
            _C['n_kinds'] = n_kinds
            _f = 1
            if fmt == 'angstrom':
                _f = constants.l_aa2au
            _C['data'] = np.array(data) * _f

            out = cls()
            out.__dict__.update(_C)
            return out

        @classmethod
        def from_data(cls, symbols, pos_au, pp=None):
            if pp is None:
                pp = 'MT_BLYP'
                warnings.warn('Setting pseudopotential in CPMD ATOMS output to'
                              ' default (Troullier-Martins, BLYP)!',
                              stacklevel=2)
            # 'SG_BLYP KLEINMAN-BYLANDER'

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
                if _e in ['C', 'O', 'N', 'Cl', 'F', 'S'] and 'AEC' not in pp:
                    # --- ToDo: replace manual tweaks by automatic pp analysis
                    channels.append("LMAX=P LOC=P")
                elif _e in ['P'] and 'AEC' not in pp:
                    channels.append("LMAX=D LOC=D")
                else:
                    channels.append("LMAX=S LOC=S")

            _C = {}
            _C['pp'] = pp
            _C['kinds'] = kinds
            _C['channels'] = channels
            _C['n_kinds'] = n_kinds
            _C['data'] = np.array(data)

            out = cls()
            out.__dict__.update(_C)
            return out


class CPMDjob():
    '''An actual representative of CPMDinput
       BETA (work in progress...)
       '''

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
    def read_input_file(cls, fn, **kwargs):
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

        CONTENT = {_C: getattr(CPMDinput, _C)._parse_section_input(CONTENT[_C],
                                                                   **kwargs)
                   for _C in CONTENT}

        return cls(**CONTENT)

    def write_input_file(self, fn, **kwargs):
        ''' CPMD 4 '''

        # known sections and order
        _SEC = ['INFO', 'CPMD', 'RESP', 'DFT', 'SYSTEM', 'ATOMS']

        if os.path.exists(fn):
            os.remove(fn)

        for _s in _SEC:
            if hasattr(self, _s):
                getattr(self, _s).write_section(
                                            fn,
                                            append=True,
                                            fmt=kwargs.get('fmt', 'angstrom'))

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
