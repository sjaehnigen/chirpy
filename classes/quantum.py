#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------

import numpy as _np
import copy
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter, minimum_filter, \
        gaussian_filter
from scipy import signal as _signal
import warnings as _warnings

from .core import _CORE
from .volume import ScalarField as _ScalarField
from .volume import VectorField as _VectorField
from .domain import Domain3D as _Domain3D
from ..physics import constants
from ..read.coordinates import xyzReader as _xyzReader


class WaveFunction(_ScalarField):
    pass


class WannierFunction(_ScalarField):
    def auto_crop(self, **kwargs):
        '''crop after threshold (default: ...)'''
        thresh = kwargs.get('thresh', 1.0)
        a = _np.amin(_np.array(self.data.shape) -
                     _np.argwhere(_np.abs(self.data) > thresh))
        b = _np.amin(_np.argwhere(_np.abs(self.data) > thresh))
        self.crop(min(a, b))

    def extrema(self, **kwargs):
        data = gaussian_filter(self.data, 4.0)
        neighborhood = generate_binary_structure(3, 3)
        local_max = maximum_filter(data, footprint=neighborhood) == data
        local_min = minimum_filter(data, footprint=neighborhood) == data
        background = (_np.abs(data) > 0.8 * _np.amax(_np.abs(data)))
        eroded_background = binary_erosion(background,
                                           structure=neighborhood,
                                           border_value=1)
        local_extreme = (local_max + local_min) * eroded_background
        return self.pos_grid()[:, local_extreme].swapaxes(0, 1)

#    @classmethod
#    def centre(cls,fn,**kwargs):
#        wfn = cls(fn=fn,**kwargs)
#        return ....#enter math here


class ElectronDensity(_ScalarField):
    def integral(self):
        # self.n_electrons = self.voxel*simps(simps(simps(self.data)))
        self.n_electrons = self.voxel*self.data.sum()
        self.threshold = 1.E-3

    def aim(self, verbose=True):
        '''Min Yu and Dallas R. Trinkle, Accurate and efficient algorithm for
           Bader charge integration, J. Chem. Phys. 134, 064111 (2011)'''
        def pbc(a, dim):
            return _np.remainder(a, self.data.shape[dim])

        def env_basin(f, x, y, z):
            return _np.array([
                        f[x,           y,           z],
                        f[pbc(x+1, 0), y,           z],
                        f[x-1,         y,           z],
                        f[x,           pbc(y+1, 1), z],
                        f[x,           y-1,         z],
                        f[x,           y,           pbc(z+1, 2)],
                        f[x,           y,           z-1]
                        ])

        self.aim_threshold = self.threshold
        boundary_max = 0
        boundary_max = max(boundary_max, _np.amax(self.data[0, :, :]))
        boundary_max = max(boundary_max, _np.amax(self.data[-1, :, :]))
        boundary_max = max(boundary_max, _np.amax(self.data[:, 0, :]))
        boundary_max = max(boundary_max, _np.amax(self.data[:, -1, :]))
        boundary_max = max(boundary_max, _np.amax(self.data[:, :, 0]))
        boundary_max = max(boundary_max, _np.amax(self.data[:, :, -1]))
        with _warnings.catch_warnings():
            if boundary_max >= self.aim_threshold:
                _warnings.warn('Density at the boundary exceeds given density '
                               'threshold of %f! %f' % (self.aim_threshold,
                                                        boundary_max),
                               RuntimeWarning,
                               stacklevel=2)

        # neighborhood = generate_binary_structure(3,1)
        test = _np.unravel_index(_np.argsort(self.data.ravel())[::-1],
                                 self.data.shape)

        atoms = range(self.n_atoms)

        basin = _np.zeros([j for i in (self.data.shape, len(atoms))
                          for j in (i if isinstance(i, tuple) else (i,))])
        atoms = iter(atoms)
        n_points = (self.data > self.aim_threshold).sum()
        g0 = self.data
        g1 = _np.roll(self.data, -1, axis=0)
        g2 = _np.roll(self.data, +1, axis=0)
        g3 = _np.roll(self.data, -1, axis=1)
        g4 = _np.roll(self.data, +1, axis=1)
        g5 = _np.roll(self.data, -1, axis=2)
        g6 = _np.roll(self.data, +1, axis=2)
        R = _np.array([g0, g1, g2, g3, g4, g5, g6]) - self.data[None]
        R[R < 0] = 0
        # R /= R.sum(axis=0)

        gain = R.sum(axis=0)
        for i in range(n_points):
            if (100*i/n_points) % 1 == 0 and verbose:
                print('Scanning point %d/%d' % (i, n_points))
            ix = test[0][i]
            iy = test[1][i]
            iz = test[2][i]
            if self.data[ix, iy, iz] > self.aim_threshold:
                if gain[ix, iy, iz] != 0:
                    basin[ix, iy, iz] = (env_basin(basin, ix, iy, iz)
                                         * R[:, ix, iy, iz, None]
                                         ).sum(axis=0)/gain[ix, iy, iz]
                else:
                    iatom = next(atoms)
                    basin[ix, iy, iz, iatom] = 1.0
        if verbose:
            print('AIM analysis done.                                              \
                    ')

        aim_atoms = list()
        pos_grid = self.pos_grid()
        for iatom in range(self.n_atoms):
            ind = _np.unravel_index(
                    _np.argmin(
                        _np.linalg.norm(
                            pos_grid[:, :, :, :] -
                            self.pos_au[iatom, :, None, None, None],
                            axis=0)), self.data.shape)
            jatom = _np.argmax(basin[ind])
            transfer = (self.comments,
                        [self.numbers[iatom]],
                        self.pos_au[iatom].reshape((1, 3)),
                        self.cell_vec_au, self.origin_au)
            # outer class won't be accessible even with inheritance
            aim_atoms.append(AIMAtom(basin[:, :, :, jatom], transfer))
        self.aim_atoms = _np.array(aim_atoms)

    @staticmethod
    def integrate_volume(data):
        # return simps(simps(simps(data)))
        return data.sum()


class AIMAtom(_Domain3D):
    def __init__(self, basin, transfer, **kwargs):
        self.comments,\
            self.numbers,\
            self.pos_au,\
            self.cell_vec_au,\
            self.origin_au = transfer
        self.grid_shape = basin.shape
        self.indices = _np.where(basin != 0)
        self.weights = basin[self.indices]

    def attach_gain_and_loss(self, gain, loss, **kwargs):
        self.j_grid_shape = gain.shape
        self.gain_indices = _np.where(gain != 0)
        self.loss_indices = _np.where(loss != 0)
        self.gain_weights = gain[self.gain_indices]
        self.loss_weights = loss[self.loss_indices]

    def expand_gain_and_loss(self):
        tmp_gain = _np.zeros(self.j_grid_shape)
        tmp_loss = _np.zeros(self.j_grid_shape)
        tmp_gain[self.gain_indices] = self.gain_weights
        tmp_loss[self.loss_indices] = self.loss_weights
        return tmp_gain, tmp_loss

    def charge(self, rho):
        def func(inds):
            return rho.data[inds] * rho.voxel
        return self.numbers.sum() - self.integrate_volume(func)


class CurrentDensity(_VectorField):
    pass


class ElectronicState(_CORE):
    def __init__(self, fn, fn1, fn2, fn3, **kwargs):
        self.psi = WaveFunction(fn, **kwargs)
        self.j = CurrentDensity(fn1, fn2, fn3, **kwargs)
        self.psi.integral()
        if not self.psi._is_similar(self.j, strict=2, return_false=True):
            raise ValueError('Wavefunction and Current are not consistent!')

    def grid(self):
        return self.psi.grid()

    def pos_grid(self):
        return self.psi.pos_grid()

    def crop(self, r, **kwargs):
        self.psi.crop(r)
        self.psi.integral()
        self.j.crop(r)

    def calculate_velocity_field(self, rho, **kwargs):
        '''Requires total density rho'''
        self.v = _VectorField.from_object(self.j)
        self.v.normalise(norm=rho.data, **kwargs)

        self.v.__class__.__name__ = "VelocityField"


class ElectronicSystem(_CORE):
    def __init__(self, fn, fn1, fn2, fn3, **kwargs):
        self.rho = ElectronDensity(fn, **kwargs)
        self.j = CurrentDensity(fn1, fn2, fn3, **kwargs)
        self.rho.integral()
        if not self.rho._is_similar(self.j, strict=2, return_false=True):
            raise ValueError('Density and Current are not consistent!')

    def grid(self):
        return self.rho.grid()

    def pos_grid(self):
        return self.rho.pos_grid()

    def crop(self, r, **kwargs):
        self.rho.crop(r)
        self.rho.integral()
        self.j.crop(r)

    def auto_crop(self, **kwargs):
        '''crop after threshold (default: ...)'''
        thresh = kwargs.get('thresh', self.rho.threshold)
        scale = self.rho.data
        a = _np.amin(_np.array(self.rho.data.shape)
                     - _np.argwhere(scale > thresh))
        b = _np.amin(_np.argwhere(scale > thresh))
        self.crop(min(a, b))

        return min(a, b)

    def calculate_velocity_field(self, **kwargs):
        self.v = _VectorField.from_object(self.j)
        self.v.normalise(norm=self.rho.data, **kwargs)

        self.v.__class__.__name__ = "VelocityField"

    def propagate_density(self, dt=8.0):
        '''dt in atomic units'''
        rho2 = copy.deepcopy(self.rho)

        self.j.divergence_and_rotation()
        rho2.data -= self.j.div * dt

        return rho2

    def read_nuclear_velocities(self, fn):
        '''Has to be in shape n_frames,n_atoms, 3'''
        self.nuc_vel_au,\
            self.nuc_symbols,\
            self.nuc_vel_comments = _xyzReader(fn)
        if list(self.nuc_symbols) != \
                constants.numbers_to_symbols(self.rho.numbers):
            raise Exception('Nuclear velocity file does not match '
                            'Electronic System!')

    def calculate_aim_differential_current(self):
        '''Map vector of nuclear velocity on atom domain'''
        self.v_diff = copy.deepcopy(self.v)
        for i in range(self.rho.n_atoms):
            field = self.rho.aim_atoms[i].map_vector(self.nuc_vel_au[0, i, :])
            self.v_diff.data -= field
            # self.rho.aim_atoms[i].map_vector(self.nuc_vel_au[0,i,:])
        self.v_diff.data = _signal.medfilt(self.v_diff.data, kernel_size=3)
        self.j_diff = copy.deepcopy(self.j)
        self.j_diff.data = self.v_diff.data*self.rho.data[_np.newaxis]

    def calculate_interatomic_flux(self, rho_dt, sign_dt):
        '''rho_dt ... ElectronicDensity object; needs aim_atoms list'''
        def rec_grid(grid):
            r_grid = _np.zeros(grid.shape)
            r_grid[grid != 0] = _np.reciprocal(grid[grid != 0])
            return r_grid
        j_norm = _np.linalg.norm(self.j.data, axis=0)
        r_j_norm = rec_grid(j_norm)
        j_dir = self.j.data*r_j_norm[None, :, :, :]*sign_dt

        j_gain = _np.zeros(self.j.data.shape)
        j_loss = _np.zeros(self.j.data.shape)

        # expand all AIM and get diff in all cartesian directions
        for i_atom, (aim_0, aim_t) in enumerate(zip(self.rho.aim_atoms,
                                                    rho_dt.aim_atoms)):
            tmp1 = aim_0.expand()
            tmp2 = aim_t.expand()
            dw = list()
            dw.append(_np.roll(tmp2, -1, axis=0))
            dw.append(_np.roll(tmp2, +1, axis=0))
            dw.append(_np.roll(tmp2, -1, axis=1))
            dw.append(_np.roll(tmp2, +1, axis=1))
            dw.append(_np.roll(tmp2, -1, axis=2))
            dw.append(_np.roll(tmp2, +1, axis=2))
            dw = _np.array(dw)-tmp1

            # gain
            gn = _np.zeros(self.j.data.shape)
            gn[0] += j_dir[0] * (j_dir[0, :, :, :] > 0) * \
                dw[0, :, :, :] * (dw[0, :, :, :] > 0)
            gn[0] += j_dir[0] * (j_dir[0, :, :, :] < 0) * \
                dw[1, :, :, :] * (dw[1, :, :, :] > 0)
            gn[1] += j_dir[1] * (j_dir[1, :, :, :] > 0) * \
                dw[2, :, :, :] * (dw[2, :, :, :] > 0)
            gn[1] += j_dir[1] * (j_dir[1, :, :, :] < 0) * \
                dw[3, :, :, :] * (dw[3, :, :, :] > 0)
            gn[2] += j_dir[2] * (j_dir[2, :, :, :] > 0) * \
                dw[4, :, :, :] * (dw[4, :, :, :] > 0)
            gn[2] += j_dir[2] * (j_dir[2, :, :, :] < 0) * \
                dw[5, :, :, :] * (dw[5, :, :, :] > 0)

            # loss
            ls = _np.zeros(self.j.data.shape)
            ls[0] += j_dir[0] * (j_dir[0, :, :, :] > 0) * \
                dw[0, :, :, :] * (dw[0, :, :, :] < 0)
            ls[0] += j_dir[0] * (j_dir[0, :, :, :] < 0) * \
                dw[1, :, :, :] * (dw[1, :, :, :] < 0)
            ls[1] += j_dir[1] * (j_dir[1, :, :, :] > 0) * \
                dw[2, :, :, :] * (dw[2, :, :, :] < 0)
            ls[1] += j_dir[1] * (j_dir[1, :, :, :] < 0) * \
                dw[3, :, :, :] * (dw[3, :, :, :] < 0)
            ls[2] += j_dir[2] * (j_dir[2, :, :, :] > 0) * \
                dw[4, :, :, :] * (dw[4, :, :, :] < 0)
            ls[2] += j_dir[2] * (j_dir[2, :, :, :] < 0) * \
                dw[5, :, :, :] * (dw[5, :, :, :] < 0)

            gn *= j_norm[None]
            ls *= j_norm[None]
            j_gain += gn
            j_loss += ls

            aim_0.attach_gain_and_loss(gn, ls)
        del tmp1, tmp2, dw, gn, ls, j_dir

        if not _np.allclose(j_gain, -j_loss, atol=1.E-4):
            with _warnings.catch_warnings():
                _warnings.warn('AIM gain/loss unbalanced!',
                               RuntimeWarning, stacklevel=2)
        print('AIM gain/loss calculation done.')

        r_j_gain = rec_grid(j_gain)
        r_j_loss = rec_grid(j_loss)

        ia_gain = _np.zeros((self.j.n_atoms, self.j.n_atoms))
        ia_loss = _np.zeros((self.j.n_atoms, self.j.n_atoms))

        z = 1
        for i, i_aim in enumerate(self.rho.aim_atoms):
            for k, k_aim in enumerate(self.rho.aim_atoms):
                # insert sparsity: only atoms that are neighboured
                # (insert check method in AIMAtom class)
                # if are_neighbours(i_aim,k_aim):
                i_gain, i_loss = i_aim.expand_gain_and_loss()
                k_gain, k_loss = k_aim.expand_gain_and_loss()
                # dt?
                ia_gain[i, k] = _np.linalg.norm(
                        (i_gain*k_loss*r_j_loss).sum(axis=(1, 2, 3)))
                ia_loss[i, k] = _np.linalg.norm(
                        (i_loss*k_gain*r_j_gain).sum(axis=(1, 2, 3)))
                # debug sign
                # else: print('Skipping')
                z += 1
        del r_j_gain, r_j_loss, i_gain, i_loss, k_gain, k_loss,\
            j_gain, j_loss, j_norm, r_j_norm

        ia_flux = ia_gain-ia_loss
        self.ia_gain = ia_gain
        self.ia_loss = ia_loss
        self.ia_flux = ia_flux
        print('IA Flux Calculation done.')

    @staticmethod
    def time_integral(dt, n='auto', n_thresh=0.1):
        '''propagates rho by dt n times and calculates gain,loss,and balance
           between atoms at each step
           n=auto ... propagate until interatomic flux (i.e., the norm of the
           balance matrix) vanishes (n_thresh)'''
        # keep also a copy of all dt's aim atoms and the latest rho
        # does not write into class but returns traj
        pass
