#!/usr/bin/env python

import argparse
import sys
import copy
import numpy as np
import time

from chirpy.classes import volume,trajectory,field
from chirpy.physics import constants
from chirpy.snippets import extract_keys

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
        help="Charge current vectorfield as cube files (fn_x, fn_y, fn_z)."
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
        '-v',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="The velocity of the reference frame in a.u. Non-zero value requires the charge \
        density unless set argument --nuclei_only. "
        )

    parser.add_argument(
        "--density",
        help="Charge density as cube file (only for moving reference; see -v option).",
        default=None
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

    parser.add_argument(
       '--verbose',
       action="store_true",
       default=False,
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

    ref_vel_au = np.array(args.v)
    if np.linalg.norm(ref_vel_au) != 0.0 and not args.nuclei_only:
        if args.density is None:
            raise AttributeError('Please set the --density argument for moving reference frame!')

    _S = list()
    _S.append(time.time())

    #-------- Read input and crop --------------
    j = volume.VectorField(*args.fn)
    if args.j_crop is not None:
        j.crop(args.j_crop)
    j.sparsity(args.j_sparse)
    j.print_info()

    if np.linalg.norm(ref_vel_au) != 0.0 and not args.nuclei_only:
        rho = volume.ScalarField(args.density)
        rho._is_similar(j, strict=2)
        j.data -= ref_vel_au[:, None, None, None] * rho.data[None]
        del rho

    #----------- Get vel and Q for atoms -------- 
    if not args.electrons_only:
        try:
            nuc = trajectory.XYZFrame(args.geofile, numbers=j.numbers, fmt='cpmd',
                    filetype='GEOMETRY')
            if not np.allclose(j.pos_au, nuc.pos_aa*constants.l_aa2au):
                raise ValueError("The given cube files do not correspond to %s!" % args.geofile)
            #Compatibility warning: In CPMD xyz velocities are given in aa per t_au! Multiplication by l_aa2au compulsory!
            #Or use CPMD format
            vel_au = nuc.vel_au - ref_vel_au[None]#*constants.l_aa2au
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

    sys.stdout.flush()
    _S.append(time.time())
    #------- Get R and crop again -------------- 
    if args.r is None:
       if args.mode == "grid":
           _tmp = copy.deepcopy(j)
           if args.r_crop is not None:
               _tmp.crop(args.r_crop)
           _tmp.sparsity(args.r_sparse)
           R = _tmp.pos_grid()

       if args.mode == "atom":
           R = j.pos_au.T
    #else:
    #    R = np.array(args.r).T.astype(float)

    #_npoints = np.prod(R.shape[1:])
    #if args.mode == "grid":
    #    print("No. of grid points: %d" % _npoints)

    #---------- Get B -------------------------- 
    if not args.nuclei_only:
        B1 = field.MagneticField.from_current(
                j,
                R=R,
                verbose=args.verbose,
                nprocs=args.nprocs,
                kspace=args.kspace
                )

    if not args.electrons_only:
        B2 = field.MagneticField.from_moving_point_charges(
                j.pos_au,
                vel_au,
                Q,
                R=R,
                **extract_keys(vars(j), cell_vec_au=None, origin_au=None)
                )

    _S.append(time.time())

    #------- Output --------- ------------------
    print( "Writing output...")

    if args.mode == 'grid':
        if not args.nuclei_only:
            B1.write(
                "B_e-1.cube", "B_e-2.cube", "B_e-3.cube",
                comment1 = "B field induced by electron current",
            )

        if not args.electrons_only:
            B2.write(
                "B_n-1.cube", "B_n-2.cube", "B_n-3.cube",
                comment1 = "B field induced by nuclear motion",
            )

        if not any([args.nuclei_only, args.electrons_only]):
            B1.data += B2.data
            B1.write(
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
        for _i, (_r,_b1,_b2) in enumerate(zip(R.T, B1.data.T, B2.data.T)):
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
