#!/usr/bin/env python

import argparse
import sys
import copy
import numpy as np
import time
from multiprocessing import Manager, Process

from chirpy.classes import volume,trajectory
from chirpy.physics.classical_electrodynamics import biot_savart, biot_savart_grid, biot_savart_kspace
from chirpy.physics import constants

def main():
    if int(np.version.version.split('.')[1]) < 14:
        print('ERROR: You have to use a numpy version >= 1.14.0! You are using %s.'%np.version.version)
        sys.exit(1)

    parser=argparse.ArgumentParser(
        description="Calculate the induced magnetic field from an electron current in real space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "fn",
        nargs=3,
        help="Electron current vectorfield as cube files (fn_x, fn_y, fn_z)."
        )

    parser.add_argument(
        '--geofile',
        default='GEOMETRY',
        help="Name of GEOMETRY file containing nuclear positions and velocities (in CPMD format)."
        )

    parser.add_argument(
        '--mode',
        default='grid',
        help="Calculate B at atom/grid positions."
        )

    parser.add_argument(
        '--nprocs',
        type=int,
        default=1,
        help="Number of parallel threads."
        )

    parser.add_argument(
        '-r',
        type=list,
        nargs="+",
        help="List of reference points (optional; overrides -mode)."
        )

    parser.add_argument(
       '--use_valence_charges',
       action='store_true',
       default=False,
       help="Calculate the nuclear contribution to the B field using atomic valence charges only (e.g., when using non-local pseudopotentials)."
       )
    parser.add_argument(
       '--j_sparse',
       type=int,
       default=1,
       help="Sparsity of j field."
       )

    parser.add_argument(
       '--r_sparse',
       type=int,
       default=1,
       help="Sparsity of position grid (grid mode only; multiplies with j_sparse)."
       )

    parser.add_argument(
       '--j_crop',
       type=int,
       default=None,
       help="Crop outer n points of j field."
       )

    parser.add_argument(
       '--r_crop',
       type=int,
       default=None,
       help="Crop outer n points of position grid (grid mode only; adds to j_crop)."
       )

    parser.add_argument(
       '--kspace',
       action="store_true",
       default=False,
       help="Do calculation in reciprocal space (ignores -mode/-nprocs/-r/--r* argument)."
       )

    parser.add_argument(
       '--nuclei_only',
       action="store_true",
       default=False,
       help="Calculate only the nuclear contribution to the B field."
       )

    parser.add_argument(
       '--electrons_only',
       action="store_true",
       default=False,
       help="Calculate only the electronic contribution to the B field."
       )

    args = parser.parse_args()

    _attr = {
        'mode' : ['atom','grid']
    }

    if args.r is not None:
        raise NotImplementedError('Choose between grid/atom mode instead of using "-r"!')

    if args.kspace:
        args.mode = "grid"
        print( 'WARNING: k space calculation still in alpha state (and no nuclear B field component, either)!' )

    for _k in _attr:
        if getattr(args,_k) not in _attr[_k]: raise AttributeError('Wrong argument for %s!' % _k)

    _S = list()
    _S.append(time.time())

    #-------- Read input -----------------------
    j = volume.VectorField(*args.fn)
    if args.j_crop is not None:
        j.crop(args.j_crop)
    j.sparsity(args.j_sparse)
    j.print_info()

    if not args.electrons_only:
        try:
            nuc = trajectory.XYZFrame(args.geofile, numbers=j.numbers, fmt='cpmd',
                    filetype='GEOMETRY')
            if not np.allclose(j.pos_au, nuc.pos_aa*constants.l_aa2au):
                raise ValueError("The given cube files do not correspond to %s!" % args.geofile)
            #Compatibility warning: In CPMD xyz velocities are given in aa per t_au! Multiplication by l_aa2au compulsory!
            #Or use CPMD format
            vel_au = nuc.vel_au#*constants.l_aa2au
            if args.use_valence_charges:
                _key = "ZV"
            else:
                _key = "Z"
            Q = [constants.species[_s][_key] for _s in nuc.symbols]
            print('Nuclear Velocities/Charges')
            print(77 * '–')
            print( '%4s '%'#' + ' '.join( "%12s" % _s for _s in ['v_x', 'v_y', 'v_z', 'Q']))
            for _i, (_v, _q) in enumerate(zip(vel_au, Q)):
                print( '%4d '%_i 
                    + ' '.join(map('{:12.5g}'.format, _v)) + ' '
                    + '{:12.5g}'.format(_q)
                    )
            print(77 * '–')
        except FileNotFoundError:
            print('WARNING: %s not found! Proceeding with electrons only.')
            args.electrons_only=True
    #------------------------------------------- 

    #-------- Create object copy for B output
    if args.mode == "grid":
        _tmp = copy.deepcopy(j)

    if args.kspace:
        print('Calculating B field in kspace...' )
        sys.stdout.flush()
        _S.append(time.time())
        B1 = biot_savart_kspace(j.data, j.cell_au, j.voxel)

    else:
        #------------------------------------------- 
        #     Multiprocessing
        #------------------------------------------- 

        #------- Some definitions ------------------ 
        def _parallel_f(R, B, n):
            B[n] = biot_savart_grid(R, j.data, j.pos_grid(), j.voxel)

        _ip2ind = lambda ip, F: np.unravel_index(ip, F.shape[1:]) 

        def _read_vec(F, ip):
            _slc = (slice(None),) + tuple([_i for _i in _ip2ind(ip,F)])
            return F[_slc]

        def _write_vec(F, ip, V):
            _slc = (slice(None),) + tuple([_i for _i in _ip2ind(ip,F)])
            F[_slc] = V

        def _collect():
            for _ip in range(_npoints):
                _write_vec(B1, _ip, result_dict[_ip])
                del result_dict[_ip]
        #-------------------------------------------
        sys.stdout.flush()
        _S.append(time.time())

        #------- Get R ----------------------------- 
        if args.r is None:
            if args.mode == "grid":
                if args.r_crop is not None:
                    _tmp.crop(args.r_crop)
                _tmp.sparsity(args.r_sparse)
                R = _tmp.pos_grid()

            if args.mode == "atom":
                R = j.pos_au.T
        #else:
        #    R = np.array(args.r).T.astype(float)

        _npoints = np.prod(R.shape[1:])
        if args.mode == "grid":
            print("No. of grid points: %d" % _npoints)

        #------- Initialise B --------------------- 
        B1 = np.zeros_like(R) 
        B2 = np.zeros_like(R) 

        if not args.nuclei_only:
            print('Calculating B field on %d core(s)...' % args.nprocs )  #Could be improved in performance. Normalisation?
            #------- Start parallel process --------- 
            _n = 0
            manager = Manager()
            result_dict = manager.dict()
            jobs = list()

            for _ip in range(_npoints):
                if _ip%100 == 0 and args.mode == "grid": 
                    print(_ip, end="\r")
                    sys.stdout.flush()
                _p = Process(target=_parallel_f, args=(_read_vec(R, _ip), result_dict, _ip))
                jobs.append(_p)
                _p.start()
                _n += 1
                if _n == args.nprocs:
                    for proc in jobs:
                        proc.join()
                    _n = 0
                    jobs = list()
            for proc in jobs:
                 proc.join()

            _collect()
            #---- It is electrons rotating: charge -1 (Factor 2 already in j???)
            B1 *= -1
            print( "Done.           " )

        if not args.electrons_only:
            print('Calculating nuclear contribution to the B field...')
            if args.mode == "grid":
                #we do not need old R anymore
                R = np.array([_read_vec(R, _ip) for _ip in range(_npoints)]).T
            _tmp_B = np.zeros_like(R.T)
            for _p,_v,_q in zip(j.pos_au, vel_au, Q):
                #change thresh because closest points will explode expression
                _tmp_B += biot_savart(R.T, _p[None,:], _v[None,:]*_q, thresh=j.voxel**(1/3))
            B2 = _tmp_B.T.reshape(B2.shape)
            del _tmp_B
            print( "Done." )
    
        if args.r is not None or args.mode == "atom":
            B1 = B1.T
            B2 = B2.T
            R = R.T
    
    _S.append(time.time())
    
    
    #------- Output --------- ------------------
    print( "Writing output...")
    
    if args.mode == 'grid':
        if not args.nuclei_only:
            _tmp.data = B1
            _tmp.write(
                "B_e-1.cube", "B_e-2.cube", "B_e-3.cube",
                comment1 = "B field induced by electron current", 
            )
    
        if not args.electrons_only:
            _tmp.data = B2
            _tmp.write(
                "B_n-1.cube", "B_n-2.cube", "B_n-3.cube",
                comment1 = "B field induced by nuclear motion", 
            )
    
        if not any([args.nuclei_only, args.electrons_only]):
            _tmp.data = B1 + B2
            _tmp.write(
                "B_total-1.cube", "B_total-2.cube", "B_total-3.cube",
                comment1 = "B field induced by nuclear motion", 
            )
    
    
    elif args.r is not None or args.mode == "atom":
        print( '' )
        print( 160 * '–' )
        print( '%4s '%'#' + ' '.join( "%12s" % _s for _s in [
            'rx', 'ry', 'rz', 
            'Bx_e', 'By_e', 'Bz_e', 
            'Bx_n', 'By_n', 'Bz_n', 
            'Bx_tot', 'By_tot', 'Bz_tot' 
        ]) )
        for _i, (_r,_b1,_b2) in enumerate(zip(R, B1, B2)):
            print( '%4d '%_i
                   + ' '.join( map( '{:12.5g}'.format, _r) ) + ' '
                   + ' '.join( map( '{:12.5g}'.format, _b1) ) + ' '
                   + ' '.join( map( '{:12.5g}'.format, _b2) ) + ' '
                   + ' '.join( map( '{:12.5g}'.format, _b1 + _b2) )
            ) #simple, only for orthorhombic
        print( 160 * '–' )
        print( '' )
    
    _S.append(time.time())
    print( '' )
    print( 77 * '–' )
    print( "Timings in seconds" )
    print( 77 * '-' )
    
    print('\n'.join( [ 
        '%-20s %10.5f' % _t for _t in zip(
            ["Reading input" ,"B field calculation" ,"Writing output" ],
            np.diff(_S) #[ _s - _S[0] for _s in _S[1:] ]
        )
    ]))
    print( 77 * '–' )
    print('')

if __name__ == "__main__":
    main()
