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

import numpy as _np
import tempfile
import warnings as _warnings

from .core import _CORE
from .volume import ScalarField as _ScalarField

# ToDo: clean up


class Domain3D(_CORE):
    '''Contains arrays of positions in a grid with assigned (scalar) values.
       The object can be expanded into a full grid representation (see volume
       class)'''

    def __init__(self,  shape,  indices,  weights,  **kwargs):
        self.grid_shape = shape
        self.indices = indices
        self.weights = weights

    def map_vector(self, v3):
        n_x, n_y, n_z = self.grid_shape
        v3_field = _np.zeros((3, n_x, n_y, n_z))
        ind = self.indices
        v3_field[:, ind[0], ind[1], ind[2]] = self.weights[None, :]*v3[:, None]
        return v3_field

    def integrate_volume(self, f):
        # return simps(f(self.indices)*self.weights)
        return _np.sum(f(self.indices)*self.weights, axis=0)

    def expand(self):
        data = _np.zeros(self.grid_shape)
        data[self.indices] = self.weights
        return data

    def write(self, fn, **kwargs):
        _ScalarField.from_domain(self, **vars(self)).write(fn, **kwargs)


class DomainSet():
    '''A collection of complimentary domains.'''

    def __init__(self, domains, **kwargs):
        self.n_domains = len(domains)
        self.shape = domains[0].grid_shape
        self.domains = domains

    def boundaries(self):
        pass  # Calculate domain boundaries/ias

#     @staticmethod
#     def _finite_differences(grid1, grid2, shape):
#         if grid1.shape!=grid2.shape: raise Exception('ERROR: Different grid shapes!')
#         FD = _np.zeros((6, )+grid1.shape)
# 
#         FD[0] = _np.roll(grid2, -1, axis=0)-grid1
#         FD[1] = _np.roll(grid2, +1, axis=0)-grid1
#         FD[2] = _np.roll(grid2, -1, axis=1)-grid1
#         FD[3] = _np.roll(grid2, +1, axis=1)-grid1
#         FD[4] = _np.roll(grid2, -1, axis=2)-grid1
#         FD[5] = _np.roll(grid2, +1, axis=2)-grid1
#         
#         self.fd = _np.zeros((self.n_domains, 6, )+self.shape)
# 
#         for i_d, (d1, d2) in enumerate(zip(domains1, domains2)):
#             #if not all([isinstance(d1, (Domain3d, Domain2D)), isinstance(d2, (Domain3d, Domain2D))]): raise Exception('ERROR: Lists contain unknown domain types!')
#             #print((type(d1).__bases__[0]))
#             #print(issubclass(type(d1).__bases__[0], classes.domain.Domain3D))
#             #if not all([issubclass(type(d1).__bases__[0], (Domain3D)), issubclass(type(d2).__bases__[0], (Domain3D))]): raise Exception('ERROR: Lists contain unknown domain types!', type(d1), type(d2))
#             if d1.grid_shape != d2.grid_shape: raise Exception('ERROR: Domain differ in their respective grid shapes!')
#             tmp1, tmp2 = d1.expand(), d2.expand()
#             #this is a gradient
#             self.fd[i_d, 0] = _np.roll(tmp2, -1, axis=0)-tmp1
#             self.fd[i_d, 1] = _np.roll(tmp2, +1, axis=0)-tmp1
#             self.fd[i_d, 2] = _np.roll(tmp2, -1, axis=1)-tmp1
#             self.fd[i_d, 3] = _np.roll(tmp2, +1, axis=1)-tmp1
#             self.fd[i_d, 4] = _np.roll(tmp2, -1, axis=2)-tmp1
#             self.fd[i_d, 5] = _np.roll(tmp2, +1, axis=2)-tmp1


class FD_Domain():
    def __init__(self, domains1, domains2, **kwargs):
        '''domains1/2 are a list of domains that must all be consistent'''
        self.origin_au = kwargs.get('origin_au', _np.array([0.0, 0.0, 0.0]))
        self.cell_au = kwargs.get('cell_au', _np.empty((0)))
        if any([self.cell_au.size == 0]):
            raise AttributeError('Please give cell_au!')

        self.voxel = _np.dot(self.cell_au[0],
                             _np.cross(self.cell_au[1], self.cell_au[2]))
        self.n_domains = len(domains1)
        self.shape = domains1[0].grid_shape
        self._tmp = kwargs.get('use_tempfile', False)
        if self.n_domains != len(domains2):
            raise ValueError('The two lists contain different numbers of domains!')
        self.fd = _np.zeros((self.n_domains, 6, )+self.shape)

        for i_d, (d1, d2) in enumerate(zip(domains1, domains2)):
            # if not all([isinstance(d1, (Domain3d, Domain2D)), isinstance(d2, (Domain3d, Domain2D))]): raise Exception('ERROR: Lists contain unknown domain types!')
            # print((type(d1).__bases__[0]))
            # print(issubclass(type(d1).__bases__[0], classes.domain.Domain3D))
            # if not all([issubclass(type(d1).__bases__[0], (Domain3D)), issubclass(type(d2).__bases__[0], (Domain3D))]): raise Exception('ERROR: Lists contain unknown domain types!', type(d1), type(d2))
            if d1.grid_shape != d2.grid_shape:
                raise ValueError('Domain differ in their respective grid shapes!')
            tmp1, tmp2 = d1.expand(), d2.expand()
            # this is a gradient
            self.fd[i_d, 0] = _np.roll(tmp2, -1, axis=0) - tmp1
            self.fd[i_d, 1] = _np.roll(tmp2, +1, axis=0) - tmp1
            self.fd[i_d, 2] = _np.roll(tmp2, -1, axis=1) - tmp1
            self.fd[i_d, 3] = _np.roll(tmp2, +1, axis=1) - tmp1
            self.fd[i_d, 4] = _np.roll(tmp2, -1, axis=2) - tmp1
            self.fd[i_d, 5] = _np.roll(tmp2, +1, axis=2) - tmp1

#    def _init_tempfile(self, name, shape):
#        try: fn = tempfile.NamedTemporaryFile(dir='/scratch/ssd/')
#        except (NotADirectoryError, FileNotFoundError): fn = tempfile.NamedTemporaryFile(dir='/tmp/')
#        setattr(self, name, _np.memmap(fn, dtype='float64', mode='w+', shape=shape))

    @staticmethod
    def normalise(v, **kwargs):  # export it (no class method)
        axis = kwargs.get('axis', 0)
        norm = kwargs.get('norm', _np.linalg.norm(v, axis=axis))
        v_dir = v/norm[None]  # only valid for axis==0
        v_dir[_np.isnan(v_dir)] = 0.0
        return v_dir, norm

    def map_flux_density(self, j):
        self.gain = _np.zeros((self.n_domains, 3, )+self.shape)
        self.loss = _np.zeros((self.n_domains, 3, )+self.shape)

        j_dir, j_norm = self.normalise(j)

        for i_d in range(self.n_domains):
            # if j_x/_y/j_z >= my_thresh
            # these are the scalar products
            self.gain[i_d, 0] += j_dir[0] * (j_dir[0] > 0) * \
                self.fd[i_d, 0] * (self.fd[i_d, 0] > 0)
            self.gain[i_d, 0] += j_dir[0] * (j_dir[0] < 0) * \
                self.fd[i_d, 1] * (self.fd[i_d, 1] > 0)
            self.gain[i_d, 1] += j_dir[1] * (j_dir[1] > 0) * \
                self.fd[i_d, 2] * (self.fd[i_d, 2] > 0)
            self.gain[i_d, 1] += j_dir[1] * (j_dir[1] < 0) * \
                self.fd[i_d, 3] * (self.fd[i_d, 3] > 0)
            self.gain[i_d, 2] += j_dir[2] * (j_dir[2] > 0) * \
                self.fd[i_d, 4] * (self.fd[i_d, 4] > 0)
            self.gain[i_d, 2] += j_dir[2] * (j_dir[2] < 0) * \
                self.fd[i_d, 5] * (self.fd[i_d, 5] > 0)

            self.loss[i_d, 0] += j_dir[0] * (j_dir[0] > 0) * \
                self.fd[i_d, 0] * (self.fd[i_d, 0] < 0)
            self.loss[i_d, 0] += j_dir[0] * (j_dir[0] < 0) * \
                self.fd[i_d, 1] * (self.fd[i_d, 1] < 0)
            self.loss[i_d, 1] += j_dir[1] * (j_dir[1] > 0) * \
                self.fd[i_d, 2] * (self.fd[i_d, 2] < 0)
            self.loss[i_d, 1] += j_dir[1] * (j_dir[1] < 0) * \
                self.fd[i_d, 3] * (self.fd[i_d, 3] < 0)
            self.loss[i_d, 2] += j_dir[2] * (j_dir[2] > 0) * \
                self.fd[i_d, 4] * (self.fd[i_d, 4] < 0)
            self.loss[i_d, 2] += j_dir[2] * (j_dir[2] < 0) * \
                self.fd[i_d, 5] * (self.fd[i_d, 5] < 0)
        self.gain *= j_norm[None, None]
        self.loss *= j_norm[None, None]

    def integrate_domains(self, **kwargs):
        flux = kwargs.get('flux', False)

        if flux:
            self.map_flux_density(kwargs.get('j'))

            fn = tempfile.TemporaryFile(dir='/scratch/ssd/')
#            try: fn = tempfile.TemporaryFile(dir='/scratch/ssd/')
#            except (NotADirectoryError, FileNotFoundError):
#                fn = tempfile.TemporaryFile(dir='/tmp/')
            transfer = _np.memmap(
                        fn,
                        dtype='float64',
                        mode='w+',
                        shape=(self.n_domains,
                               self.n_domains,
                               3) + self.shape,
                        )

            j_gain = self.gain.sum(axis=0)
            j_loss = self.loss.sum(axis=0)
            with _warnings.catch_warnings():
                if not _np.allclose(j_gain, -j_loss, atol=1.E-3):
                    _warnings.warn('Domain gain/loss unbalanced!',
                                   RuntimeWarning, stacklevel=2)

            w_gain, j_gain = self.normalise(self.gain, norm=j_gain)
            w_loss, j_loss = self.normalise(self.loss, norm=j_loss)

            transfer[:, :] = w_gain[:, None] * w_loss[None, :] * j_gain[None, None]
            transfer[:, :] -= w_loss[:, None] * w_gain[None, :] * j_loss[None, None]
            intg = _np.zeros((self.n_domains, self.n_domains, ))
            # for i_d in range(self.n_domains):
                # for j_d in range(self.n_domains):
                    # data = self.fd[i_d, j_d]
                    # intg[i_d, j_d] = _ScalarField(fmt='manual', data=data, **vars(self)).integral()
            intg = transfer.sum(axis=-1).sum(axis=-1).sum(axis=-1).sum(axis=-1)
            print(intg)

        else:
            for i_d in range(self.n_domains):
                data = self.fd[i_d]
                intg = _ScalarField(
                            fmt='manual',
                            data=data.sum(axis=0),
                            **vars(self)
                            ).integral()
                print(intg)

#            self._init_tempfile('gain', (self.n_domains, 3, )+self.shape)
#            self._init_tempfile('transfer_gain', (self.n_domains, self.n_domains, 3, )+self.shape)
#            self._init_tempfile('loss', (self.n_domains, 3, )+self.shape)
#            self._init_tempfile('transfer_loss', (self.n_domains, self.n_domains, 3, )+self.shape)
#            self.gain.fill(0.0) #unnecessary?
#            self.loss.fill(0.0) #unnecessary?
#        else: 
#            self.gain = _np.zeros((self.n_domains, 3, )+self.shape)
#            self.transfer_gain = _np.zeros((self.n_domains, self.n_domains, 3, )+self.shape)
#            self.loss = _np.zeros((self.n_domains, 3, )+self.shape)
#            self.transfer_loss = _np.zeros((self.n_domains, self.n_domains, 3, )+self.shape)
#        j_gain = self.gain.sum(axis=0)
#        j_loss = self.loss.sum(axis=0)
#        if not _np.allclose(j_gain, -j_loss, atol=1.E-3): print('WARNING: Domain gain/loss unbalanced!')
#
#        w_gain, j_gain = self.normalise(self.gain, norm=j_gain)
#        w_loss, j_loss = self.normalise(self.loss, norm=j_loss)
#    
#        self.transfer_gain = w_gain[:, None]*w_loss[None, :]*j_gain[None, None]
#        self.transfer_loss = w_loss[:, None]*w_gain[None, :]*j_loss[None, None]
#self.n_electrons = self.voxel*simps(simps(simps(self.data)))


####get target point    
####for all atoms: Calculate balance: Transfer(point1, point2, atom_i) = (w_aim2(atom_i, point2)-w_aim1(atom_i, point1))*j_x
