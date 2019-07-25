#!/usr/bin/env python

import sys
import numpy as np
from multiprocessing import Manager, Process
from chirpy.physics.classical_electrodynamics import biot_savart, biot_savart_grid, biot_savart_kspace
#import copy

from ..classes.volume import VectorField

class MagneticField(VectorField):
    def change_reference_frame():
        pass

    @classmethod
    def from_current(cls, j, **kwargs):
        '''Calculate B under the magnetostatic approximation'''
        verbose = kwargs.get('verbose', False)
        nprocs = kwargs.get('nprocs', 1)
        #---- default: electrons rotating (charge -1)
        charge = kwargs.pop('charge', -1)
        if kwargs.pop('kspace', False):
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
            R = kwargs.pop('R', j.pos_grid())
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
    def from_moving_point_charges(cls, pos_au, vel_au, Q, **kwargs):
        '''
        pos_au/vel_au...positions velocities (N,3)
        Q...charge (in e)
        R...of shape (3, n_positions)
        '''
        charge = kwargs.pop('charge', +1)
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('Calculating nuclear contribution to the B field...')

        R = kwargs.pop('R', pos_au.T)
        _shape = R.shape
        _npoints = np.prod(R.shape[1:])

        #if args.mode == "grid":
        #      #we do not need old R anymore
        #flattening
        R = np.array([cls._read_vec(R, _ip) for _ip in range(_npoints)]).T
        _tmp_B = np.zeros_like(R.T)

        for _p, _v, _q in zip(pos_au, vel_au, Q):
            #change thresh because closest points will explode expression
            _tmp_B += biot_savart(R.T, _p[None,:], _v[None,:]*_q, thresh=0.01) #j.voxel**(1/3))
        B2 = _tmp_B.T.reshape(_shape) * charge
        del _tmp_B
        if verbose:
            print( "Done." )
        return cls.from_data(data=B2, **kwargs)

class ElectricField(VectorField):
    @classmethod
    def from_charge_density(cls):
        pass
