#!/usr/bin/env python

import sys
import numpy as np
from multiprocessing import Manager, Process

from ..physics.classical_electrodynamics import biot_savart, biot_savart_grid, biot_savart_kspace
from ..physics.classical_electrodynamics import coulomb, coulomb_grid, coulomb_kspace
from ..topology.grid import map_on_posgrid
from ..classes.volume import VectorField, ScalarField

class MagneticField(VectorField):
    def change_reference_frame():
        pass

    @classmethod
    def from_current(cls, j, **kwargs):
        '''Calculate B under the magnetostatic approximation (Biot-Savart law)'''
        verbose = kwargs.get('verbose', False)
        nprocs = kwargs.get('nprocs', 1)
        #---- default: electrons moving (charge -1)
        charge = kwargs.get('charge', -1)
        if kwargs.get('kspace', False):
            sys.stdout.flush()
            B1 = biot_savart_kspace(j.data, j.cell_au, j.voxel)

        else:
            #------------------------------------------- 
            #     Multiprocessing
            #------------------------------------------- 

            #------- Some definitions ------------------ 
            def _parallel_f(R, B, n):
                B[n] = biot_savart_grid(R, j.data, j.pos_grid(), j.voxel)

            def _collect():
                for _ip in range(_npoints):
                    cls._write_vec(B1, _ip, result_dict[_ip])
                    del result_dict[_ip]


            #------- Get R -----------------------------
            R = kwargs.get('R', j.pos_grid())
            _npoints = np.prod(R.shape[1:])

            if verbose:
                print("No. of grid points: %d" % _npoints)
            if verbose:
                print('Calculating B field on %d core(s)...' % nprocs )  #Could be improved in performance. Normalisation?

            #------- Init --------------------------- 
            B1 = np.zeros_like(R)
            #------- Start parallel process --------- 
            _n = 0
            manager = Manager()
            result_dict = manager.dict()
            jobs = list()

            for _ip in range(_npoints):
                if _ip%100 == 0 and verbose:
                    print(_ip, end="\r")
                    sys.stdout.flush()
                _p = Process(target=_parallel_f, args=(cls._read_vec(R, _ip), result_dict, _ip))
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
                print( "Done.           " )
        nargs = vars(j)
        nargs.update({'data':B1})
        return cls.from_data(**nargs)


    @classmethod
    def from_moving_point_charges(cls, P_au, V_au, Q, **kwargs):
        '''
        P_au/V_au...positions velocities (N,3)
        Q...charge (in e)
        R...of shape (3, n_positions)
        '''
        charge = kwargs.get('charge', +1)
        verbose = kwargs.get('verbose', False)
        smear = kwargs.get('smear_charges', True)
        if verbose:
            print('Calculating nuclear contribution to the B field...')

        R = kwargs.get('R', P_au.T)
        if not smear:
            _shape = R.shape
            _npoints = np.prod(R.shape[1:])

            #flattening
            R = np.array([cls._read_vec(R, _ip) for _ip in range(_npoints)]).T

            _tmp_B = np.zeros_like(R.T)
            for _p, _v, _q in zip(P_au, V_au, Q):
                #change thresh because closest points will explode expression
                _tmp_B += biot_savart(R.T, _p[None,:], _v[None,:]*_q, thresh=0.01) #j.voxel**(1/3))
            B2 = _tmp_B.T.reshape(_shape) * charge
            if verbose:
                print( "Done." )
            return cls.from_data(data=B2, **kwargs)

        else:
            _cell = np.diag(R[:, 1, 1, 1]-R[:, 0, 0, 0])
            if verbose:
                print(' (using smeared point charges.)')
            #No pbc support for now (cell_aa_deg=None)
            _tmp = np.sum([
                    _v[:, None, None, None]\
                        * map_on_posgrid(_p, R, 0.2, cell_aa_deg=None, mode="gaussian", dim=3)\
                        * _q\
                        for _p, _v, _q in zip(P_au, V_au, Q)
                ], axis=0)
            _j = VectorField.from_data(data=_tmp, cell_vec_au=_cell, origin_au=R[:, 0, 0, 0])
            return cls.from_current(_j, **kwargs)


class ElectricField(VectorField):
    @classmethod
    def from_charge_density(cls, rho, **kwargs):
        '''Calculate E under the electrostatic approximation (Coulomb's law)'''
        verbose = kwargs.get('verbose', False)
        nprocs = kwargs.get('nprocs', 1)
        #---- default: electrons moving (charge -1)
        charge = kwargs.get('charge', -1)
        if kwargs.get('kspace', False):
            sys.stdout.flush()
            E1 = coulomb_kspace(rho.data, rho.cell_au, rho.voxel)

        else:
            #------------------------------------------- 
            #     Multiprocessing
            #------------------------------------------- 

            #------- Some definitions ------------------ 
            def _parallel_f(R, E, n):
                E[n] = coulomb_grid(R, rho.data, rho.pos_grid(), rho.voxel)

            def _collect():
                for _ip in range(_npoints):
                    cls._write_vec(E1, _ip, result_dict[_ip])
                    del result_dict[_ip]


            #------- Get R -----------------------------
            R = kwargs.get('R', rho.pos_grid())
            _npoints = np.prod(R.shape[1:])
            if verbose:
                print("No. of grid points: %d" % _npoints)
            if verbose:
                print('Calculating E field on %d core(s)...' % nprocs )  #Could be improved in performance. Normalisation?

            #------- Init --------------------------- 
            E1 = np.zeros_like(R)
            #------- Start parallel process --------- 
            _n = 0
            manager = Manager()
            result_dict = manager.dict()
            jobs = list()

            for _ip in range(_npoints):
                if _ip%100 == 0 and verbose:
                    print(_ip, end="\r")
                    sys.stdout.flush()
                _p = Process(target=_parallel_f, args=(cls._read_vec(R, _ip), result_dict, _ip))
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
                print( "Done.           " )
        nargs = vars(rho)
        nargs.update({'data':E1})
        return cls.from_data(**nargs)

    @classmethod
    def from_point_charges(cls, P_au, Q, **kwargs):
        '''
        P_au...positions velocities (N,3)
        Q...charge (in e)
        R...of shape (3, n_positions)
        '''
        charge = kwargs.get('charge', +1)
        verbose = kwargs.get('verbose', False)
        smear = kwargs.get('smear_charges', True)
        if verbose:
            print('Calculating nuclear contribution to the E field...')

        R = kwargs.get('R', P_au.T)
        if not smear:
            _shape = R.shape
            _npoints = np.prod(R.shape[1:])

            #flattening
            R = np.array([cls._read_vec(R, _ip) for _ip in range(_npoints)]).T
            _tmp_E = np.zeros_like(R.T)

            for _p, _q in zip(P_au, Q):
                #change thresh because closest points will explode expression
                _tmp_E += coulomb(R.T, _p[None,:], _q, thresh=0.01)
            E2 = _tmp_E.T.reshape(_shape) * charge
            if verbose:
                print( "Done." )
            return cls.from_data(data=E2, **kwargs)

        else:
            _cell = np.diag(R[:, 1, 1, 1]-R[:, 0, 0, 0])
            if verbose:
                print(' (using smeared point charges.)')
            #No pbc support for now (cell_aa_deg=None)
            _tmp = np.sum([
                        map_on_posgrid(_p, R, 0.2, cell_aa_deg=None, mode="gaussian", dim=3)\
                        * _q\
                        for _p, _q in zip(P_au, Q)
                ], axis=0)
            _rho = ScalarField.from_data(data=_tmp, cell_vec_au=_cell, origin_au=R[:, 0, 0, 0])

            return cls.from_charge_density(_rho, **kwargs)
