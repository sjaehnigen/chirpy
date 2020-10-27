# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2020, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import sys as _sys
import numpy as _np
from multiprocessing import Manager, Process

from .volume import VectorField as _VectorField
from .volume import ScalarField as _ScalarField
from ..physics.classical_electrodynamics import biot_savart as _biot_savart
from ..physics.classical_electrodynamics import biot_savart_grid \
        as _biot_savart_grid
from ..physics.classical_electrodynamics import biot_savart_kspace \
        as _biot_savart_kspace
from ..physics.classical_electrodynamics import coulomb as _coulomb
from ..physics.classical_electrodynamics import coulomb_grid as _coulomb_grid
from ..physics.classical_electrodynamics import coulomb_kspace \
        as _coulomb_kspace
from ..physics import constants
from ..topology.grid import regularisation as _regularisation


class MagneticField(_VectorField):
    def change_reference_frame():
        pass

    @classmethod
    def from_current(cls, j,
                     kspace=False, R_aa=None,
                     charge=-1, nprocs=1, verbose=False):
        '''Calculate B under the magnetostatic approximation
           (Biot-Savart law)
           '''
        # --- default: electrons moving (charge -1)
        if kspace:
            _sys.stdout.flush()
            # B1 = _biot_savart_kspace(j.data, j.cell_vec_aa, j.voxel) \
            #     * constants.l_aa2au**4
            # --- equivalent to:
            B1 = _biot_savart_kspace(j.data,
                                     j.cell_vec_aa*constants.l_aa2au,
                                     j.voxel*constants.l_aa2au**3)

        else:
            # ---
            #     Multiprocessing
            # ---

            # --- Some definitions
            def _parallel_f(R, B, n):
                B[n] = _biot_savart_grid(R*constants.l_aa2au,
                                         j.data,
                                         j.pos_grid()*constants.l_aa2au,
                                         j.voxel*constants.l_aa2au**3)

            def _collect():
                for _ip in range(_npoints):
                    cls._write_vec(B1, _ip, result_dict[_ip])
                    del result_dict[_ip]

            # --- Get R
            if R_aa is None:
                R_aa = j.pos_grid()
            _npoints = _np.prod(R_aa.shape[1:])

            if verbose:
                print("No. of grid points: %d" % _npoints)
            if verbose:
                print('Calculating B field on %d core(s)...' % nprocs)

            # --- Init
            B1 = _np.zeros_like(R_aa)
            # --- Start parallel process
            _n = 0
            manager = Manager()
            result_dict = manager.dict()
            jobs = list()

            for _ip in range(_npoints):
                if _ip % 100 == 0 and verbose:
                    # print(_ip, end="\r")
                    _sys.stdout.flush()
                _p = Process(
                        target=_parallel_f,
                        args=(cls._read_vec(R_aa, _ip), result_dict, _ip)
                        )
                jobs.append(_p)
                _p.start()
                _n += 1
                if _n == nprocs:
                    for proc in jobs:
                        proc.join()
                    _n = 0
                    jobs = list()
            for proc in jobs:
                proc.join()

            _collect()
            B1 *= charge
            if verbose:
                print("Done.           ")

        return cls.from_object(j, data=B1)

    @classmethod
    def from_moving_point_charges(cls, P_au, V_au, Q, **kwargs):
        # --- ToDo: change units to angstrom, TEST
        '''
        P_au/V_au...positions velocities (N,3)
        Q...charge (in e)
        R...of shape (3, n_positions)
        '''
        charge = kwargs.get('charge', +1)
        verbose = kwargs.get('verbose', False)
        smear = kwargs.get('smear_charges', True)
        smear_sigma = kwargs.get('smear_sigma', 0.2)
        thresh = kwargs.get('thresh', 0.01)
        if verbose:
            print('Calculating nuclear contribution to the B field...')

        R = kwargs.get('R', P_au.T)
        if not smear:
            _shape = R.shape
            _npoints = _np.prod(R.shape[1:])

            # flattening
            R = _np.array([cls._read_vec(R, _ip) for _ip in range(_npoints)]).T

            _tmp_B = _np.zeros_like(R.T)
            for _p, _v, _q in zip(P_au, V_au, Q):
                # change thresh because closest points will explode expression
                _tmp_B += _biot_savart(R.T,
                                       _p[None, :],
                                       _v[None, :] * _q,
                                       thresh=thresh)  # j.voxel**(1/3))
            B2 = _tmp_B.T.reshape(_shape) * charge
            if verbose:
                print("Done.")
            return cls.from_data(data=B2, **kwargs)

        else:
            _cell = _np.diag(R[:, 1, 1, 1]-R[:, 0, 0, 0])
            if verbose:
                print(' (using smeared point charges.)')
            # No pbc support for now (cell_aa_deg=None)
            _tmp = _np.sum(
                    V_au[:, :, None, None, None]
                    * _regularisation(P_au, R, smear_sigma,
                                      mode="gaussian")[:, None]
                    * Q[:, None, None, None, None],
                    axis=0)

            _j = _VectorField.from_data(
                                data=_tmp,
                                cell_vec_au=_cell,
                                origin_au=R[:, 0, 0, 0])
            kwargs.update({'charge': charge})
            return cls.from_current(_j, **kwargs)


class ElectricField(_VectorField):
    @classmethod
    def from_charge_density(cls, rho,
                            kspace=False, R_aa=None,
                            charge=-1, nprocs=1, verbose=False):
        '''Calculate E under the electrostatic approximation (Coulomb's law)'''
        if kspace:
            _sys.stdout.flush()
            E1 = _coulomb_kspace(rho.data,
                                 rho.cell_vec_aa*constants.l_aa2au,
                                 rho.voxel*constants.l_aa2au**3)

        else:
            # ---
            #     Multiprocessing

            # --- Some definitions
            def _parallel_f(R, E, n):
                E[n] = _coulomb_grid(R*constants.l_aa2au,
                                     rho.data,
                                     rho.pos_grid()*constants.l_aa2au,
                                     rho.voxel*constants.l_aa2au**3)

            def _collect():
                for _ip in range(_npoints):
                    cls._write_vec(E1, _ip, result_dict[_ip])
                    del result_dict[_ip]

            # --- Get R
            if R_aa is None:
                R_aa = rho.pos_grid()
            _npoints = _np.prod(R_aa.shape[1:])
            if verbose:
                print("No. of grid points: %d" % _npoints)
            if verbose:
                print('Calculating E field on %d core(s)...' % nprocs)

            # --- Init
            E1 = _np.zeros_like(R_aa)
            # --- Start parallel process
            _n = 0
            manager = Manager()
            result_dict = manager.dict()
            jobs = list()

            for _ip in range(_npoints):
                if _ip % 100 == 0 and verbose:
                    # print(_ip, end="\r")
                    _sys.stdout.flush()
                _p = Process(
                        target=_parallel_f,
                        args=(cls._read_vec(R_aa, _ip), result_dict, _ip)
                        )
                jobs.append(_p)
                _p.start()
                _n += 1
                if _n == nprocs:
                    for proc in jobs:
                        proc.join()
                    _n = 0
                    jobs = list()
            for proc in jobs:
                proc.join()

            _collect()
            E1 *= charge
            if verbose:
                print("Done.           ")

        return cls.from_object(rho, data=E1)

    @classmethod
    def from_point_charges(cls, P_au, Q, **kwargs):
        # --- ToDo: change units to angstrom, TEST
        '''
        P_au...positions (N,3)
        Q...charges (N) (in e)
        R...of shape (3, n_positions)
        '''
        charge = kwargs.get('charge', +1)
        verbose = kwargs.get('verbose', False)
        smear = kwargs.get('smear_charges', True)
        smear_sigma = kwargs.get('smear_sigma', 0.2)
        thresh = kwargs.get('thresh', 0.01)
        if verbose:
            print('Calculating nuclear contribution to the E field...')

        R = kwargs.get('R', P_au.T)
        if not smear:
            _shape = R.shape
            _npoints = _np.prod(R.shape[1:])

            # flattening
            R = _np.array([cls._read_vec(R, _ip) for _ip in range(_npoints)]).T
            _tmp_E = _np.zeros_like(R.T)

            for _p, _q in zip(P_au, Q):
                # change thresh because closest points will explode expression
                _tmp_E += _coulomb(R.T, _p[None, :], _q, thresh=thresh)
            E2 = _tmp_E.T.reshape(_shape) * charge
            if verbose:
                print("Done.")
            return cls.from_data(data=E2, **kwargs)

        else:
            _cell = _np.diag(R[:, 1, 1, 1] - R[:, 0, 0, 0])
            if verbose:
                print(' (using smeared point charges.)')
            # No pbc support for now (cell_aa_deg=None)
            _tmp = _np.sum(
                    _regularisation(P_au, R, smear_sigma, mode="gaussian")
                    * Q[:, None, None, None],
                    axis=0
                    )

            _rho = _ScalarField.from_data(data=_tmp,
                                          cell_vec_au=_cell,
                                          origin_au=R[:, 0, 0, 0])

            kwargs.update({'charge': charge})
            return cls.from_charge_density(_rho, **kwargs)
