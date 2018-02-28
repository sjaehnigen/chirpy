#!/usr/bin/env python3

import numpy as np
import argparse
import sys
import os
from classes import molecule,pymol

Angstrom2Bohr = 1.8897261247828971
#from fileio import xyz
from lib import constants
#from geometry import transformations

def main():
    parser=argparse.ArgumentParser(description="Generate Pymol script for the visualisation of vector field based on data in xyz format.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn_xyz",   help="XYZ file with starting points containing velocities")
    parser.add_argument("--scale",   help="Vels scaling factor",         type=float,    default=1000)
    parser.add_argument("--factor",  help="Plot scaling factor",         type=float,    default=1.0)
    parser.add_argument("--cutoff",   help="Cutoff (default: 0.15)",     type=float,    default=0.15)
    parser.add_argument("--head",     help="Head width (default: 0.15)", type=float,    default=0.15)
    parser.add_argument("--tail",     help="Tail width (default: 0.05)", type=float,    default=0.05)
    parser.add_argument("--draw_tail",  help="Draw tail?", action='store_true',         default=False)
    parser.add_argument("--head_length",help="Relative head length",     type=float,    default=0.7)
    parser.add_argument("--f",        help="Output file name",                          default='output.pymolrc')
    parser.add_argument("--color_rgb",nargs=3, help="Color code of the arrows (default: 0 0 0, i.e. black)",  default=(0.0,0.0,0.0))
#    parser.add_argument("--fn_vel", help="XYZ file with velocities in au (overrides values given in fn_xyz)", default=None)
#    parser.add_argument("--ts",     help="Timestep (default: 1.0 fs)",                             default=1.0)
    #parser.add_argument("--mw",     help="Mass weighted velocities. Default: False", action='store_true',              default=False)
    #parser.add_argument("--velnorm",help="Normalise velocities: maxnorm (default: False)", action='store_true',          default=False)
    args = parser.parse_args()
    rgb = tuple([float(i) for i in args.color_rgb])

    mol = molecule.Molecule(args.fn_xyz)

    p0 = mol.XYZData.pos_aa
    p1 = mol.XYZData.pos_aa + mol.XYZData.vel_au*args.scale

    ind  = np.linalg.norm(p0-p1,axis=-1) >= args.cutoff #preselection
    obj0 = pymol.PymolObject(p0[ind],
                             p1[ind],
                             name='vectors',
                             type='modevectors',
                             #headrgb=(0.0,0.0,0.0),tailrgb=(0.0,0.0,0.0),
                             headrgb=rgb,tailrgb=rgb,
                             notail=not args.draw_tail,
                             head=args.head,
                             tail=args.tail,
                             head_length=args.head_length,
                             cutoff = args.cutoff,
                             factor = args.factor) 
    obj = obj0#+obj1+obj2
    obj.write(args.f)

if(__name__ == "__main__"):
    main()



    #mw = args.mw
    #tail = args.tail
    #velnorm = args.velnorm
    #cutoff = float(args.cutoff)
#    #data, symbols, comments = xyz.ReadTrajectory_BruteForce(fn_xyz)
#    #pos_aa  = data[0:1,:,:3] #For the time being: only first frame (trajectories later)
#    #if fn_vel:
#    #    vel_au, symbols, comments = xyz.ReadTrajectory_BruteForce(fn_vel)
#    else:
#        vel_au  = data[0:1,:,3:]
#    comments = comments[0:1]
#    print(pos_aa.shape)
#    p1 = pos_aa
#    #Normalise vels 
#    if mw==True:
#        masses  =  transformations.GetMasses(symbols)
#        weight  =  np.array(masses)/sum(masses)*len(masses)
#        vel_au *=  weight[np.newaxis,:,np.newaxis]    
#    if velnorm:
#        vel_au /= np.amax(np.linalg.norm(vel_au,axis=2),axis=1)[:,np.newaxis,np.newaxis] #not the best way to use np.mean --> better norm or so
#
#    p2 = pos_aa + ts*vel_au*constants.v_au2aaperfs
#    #com_aa  =  transformations.CentersOfMass(pos_aa,masses)
#    #pos_aa  += box_aa/2 - com_aa




