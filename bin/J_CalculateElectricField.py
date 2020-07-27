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


import argparse
import sys
import copy
import numpy as np
import time

from chirpy.classes import volume, trajectory, field
from chirpy.physics import constants
from chirpy.snippets import extract_keys


def main():
    if int(np.version.version.split('.')[1]) < 14:
        print('ERROR: You have to use a numpy version >= 1.14.0! You are using %s.'%np.version.version)
        sys.exit(1)

    parser=argparse.ArgumentParser(
        description="Calculate the electric field from an electron and nuclear charge density.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "fn",
        help="Charge density as cube file.",
        default=None
        )

    parser.add_argument(
        '--geofile',
        default='GEOMETRY',
        help="Name of GEOMETRY file containing nuclear positions (in CPMD format)."
        )

    parser.add_argument(
        '--mode',
        default='grid',
        help="Calculate E at atom/grid positions."
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

#    parser.add_argument(
#        '-v',
#        type=float,
#        nargs=3,
#        default=[0.0, 0.0, 0.0],
#        help="The velocity of the reference frame in a.u. Non-zero value requires the charge \
#        density unless set argument --nuclei_only. "
#        )

    parser.add_argument(
       '--use_valence_charges',
       action='store_true',
       default=False,
       help="Calculate the nuclear contribution to the E field using atomic valence charges only (e.g., when using non-local pseudopotentials)."
       )

    parser.add_argument(
       '--q_sparse',
       type=int,
       default=1,
       help="Sparsity of charge density."
       )

    parser.add_argument(
       '--r_sparse',
       type=int,
       default=1,
       help="Sparsity of position grid (grid mode only; multiplies with q_sparse)."
       )

    parser.add_argument(
       '--q_crop',
       type=int,
       default=None,
       help="Crop outer n points of charge density."
       )

    parser.add_argument(
       '--r_crop',
       type=int,
       default=None,
       help="Crop outer n points of position grid (grid mode only; adds to q_crop)."
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
       help="Calculate only the nuclear contribution to the E field."
       )

    parser.add_argument(
       '--electrons_only',
       action="store_true",
       default=False,
       help="Calculate only the electronic contribution to the E field."
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
        raise NotImplementedError('Please use the real space calculation!')

    for _k in _attr:
        if getattr(args,_k) not in _attr[_k]: raise AttributeError('Wrong argument for %s!' % _k)

    _S = list()
    _S.append(time.time())

    #-------- Read input and crop --------------
    rho = volume.ScalarField(args.fn)

    if args.q_crop is not None:
        rho.crop(args.q_crop)
    rho = rho.sparse(args.q_sparse)
    rho.print_info()

    #----------- Get vel and Q for atoms -------- 
    if not args.electrons_only:
        try:
            nuc = trajectory.XYZFrame(args.geofile, numbers=rho.numbers, fmt='cpmd',
                    filetype='GEOMETRY')
            if not np.allclose(rho.pos_au, nuc.pos_aa*constants.l_aa2au):
                raise ValueError("The given cube files do not correspond to %s!" % args.geofile)
            #Compatibility warning: In CPMD xyz velocities are given in aa per t_au! Multiplication by l_aa2au compulsory!
            #Or use CPMD format
            pos_au = rho.pos_au
            if args.use_valence_charges:
                _key = "valence_charges"
            else:
                _key = "atomic_numbers"
            Q = np.array([getattr(constants, _key)[_s] for _s in nuc.symbols])
            print('Nuclear Positions/Charges')
            print(77 * '–')
            print( '%4s '%'#' + ' '.join( "%12s" % _s for _s in ['p_x', 'p_y', 'p_z', 'Q']))
            for _i, (_v, _q) in enumerate(zip(pos_au, Q)):
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
           _tmp = copy.deepcopy(rho)
           if args.r_crop is not None:
               _tmp.crop(args.r_crop)
           _tmp = _tmp.sparse(args.r_sparse)
           R = _tmp.pos_grid()
           smear_charges = True

       if args.mode == "atom":
           R = rho.pos_au.T
           smear_charges = False
    #else:
    #    R = np.array(args.r).T.astype(float)

    #_npoints = np.prod(R.shape[1:])
    #if args.mode == "grid":
    #    print("No. of grid points: %d" % _npoints)

    #---------- Get B -------------------------- 
    if not args.nuclei_only:
        E1 = field.ElectricField.from_charge_density(
                rho,
                R=R,
                verbose=args.verbose,
                nprocs=args.nprocs,
                kspace=args.kspace,
                )

    if not args.electrons_only:
        E2 = field.ElectricField.from_point_charges(
                pos_au,
                Q,
                R=R,
                verbose=args.verbose,
                nprocs=args.nprocs,
                kspace=args.kspace,
                smear_charges = smear_charges,
                **extract_keys(vars(rho), cell_vec_au=None, origin_au=None, numbers=None, pos_au=None)
                )

    _S.append(time.time())

    #------- Output --------- ------------------
    print( "Writing output...")

    if args.mode == 'grid':
        if not args.nuclei_only:
            E1.write(
                "E_e-1.cube", "E_e-2.cube", "E_e-3.cube",
                comment1 = 3*["E field by electron density"],
            )

        if not args.electrons_only:
            E2.write(
                "E_n-1.cube", "E_n-2.cube", "E_n-3.cube",
                comment1 = 3*["E field by nuclear charges"],
            )

        if not any([args.nuclei_only, args.electrons_only]):
            E1.data += E2.data
            E1.write(
                "E_total-1.cube", "E_total-2.cube", "E_total-3.cube",
                comment1 = 3*["E field nuclear charges and electron density"],
            )

    elif args.r is not None or args.mode == "atom":
        print( '' )
        print( 160 * '–' )
        print( '%4s '%'#' + ' '.join( "%12s" % _s for _s in [
            'rx', 'ry', 'rz',
            'Ex_e', 'Ey_e', 'Ez_e',
            'Ex_n', 'Ey_n', 'Ez_n',
            'Ex_tot', 'Ey_tot', 'Ez_tot'
        ]) )
        for _i, (_r,_b1,_b2) in enumerate(zip(R.T, E1.data.T, E2.data.T)):
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
            ["Reading input" ,"E field calculation" ,"Writing output" ],
            np.diff(_S) #[ _s - _S[0] for _s in _S[1:] ]
        )
    ]))
    print( 77 * '–' )
    print('')

if __name__ == "__main__":
    main()
