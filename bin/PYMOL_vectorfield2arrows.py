#!/usr/bin/env python

import numpy as np
import argparse
from chirpy.classes import volume,pymol

Angstrom2Bohr = 1.8897261247828971

def main():
    parser=argparse.ArgumentParser(description="Generate Pymol script for the visualisation of vector field based on data in cube format (for now only in 3D).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fn_cube",   nargs=3, help="Three cube file for three dimensions.")
    parser.add_argument("--coord",   help="Which coordinate to scan (only one necessary, 1-->x, 2-->y, 3-->z).", default=2)
    parser.add_argument("--scale",   help="Cube scaling factor",                        default=1000)
    parser.add_argument("--factor",  help="Plot scaling factor",                        default=1.0)
    parser.add_argument("--cutoff",   help="Cutoff (default: 0.15)",                    default=0.15)
    parser.add_argument("--head",     help="Head width (default: 0.15)",                default=0.15)
    parser.add_argument("--head_length",   help="Relative head length (default: 0.7)",  default=0.7)
    parser.add_argument("--sparse",  help="Use every <sparse> grid point",              default=1)
    parser.add_argument("--c0",       help="Which grid index of coord to start with",   default=0)
    parser.add_argument("--c1",       help="Which grid index of coord to finish with",  default=-1)
    parser.add_argument("--f",        help="Output file name",                          default='output.pymolrc')
    parser.add_argument("--color_rgb",nargs=3, help="Color code of the arrows (default: 0 0 0, i.e. black)",  default=(0.0,0.0,0.0))
    args = parser.parse_args()
    fn_cube = args.fn_cube
    coord   = args.coord
    segment = [args.c0,args.c1]
    scale   = float(args.scale)
    factor  = float(args.factor)
    cutoff  = float(args.cutoff)
    head    = float(args.head)
    head_length  = float(args.head_length)
    sparse  = int(args.sparse)
    rgb = tuple([float(i) for i in args.color_rgb])
    fn_out  = args.f


    vfield = volume.VectorField(fn_cube[0],fn_cube[1],fn_cube[2])
    grid   = np.moveaxis(vfield.pos_grid(),0,-1)
    flux_t = np.moveaxis(vfield.data,0,-1)*scale
#    p0, p1 =  ReadFluxFromCube(fn_cube[0],fn_cube[1],fn_cube[2],coord,segment,scale,sparse)
    i0 = segment[0]
    i1 = segment[1]
    if i1 == -1:
        i1 = vfield.data.shape[coord-1]

    
    if coord == 1:
        pos1 = grid[i0:i1:sparse,::sparse,::sparse,:]
        pos2 = grid[i0:i1:sparse,::sparse,::sparse,:]+flux_t[i0:i1:sparse,::sparse,::sparse,:]
    elif coord == 2:
        pos1 = grid[::sparse,i0:i1:sparse,::sparse]
        pos2 = grid[::sparse,i0:i1:sparse,::sparse,:]+flux_t[::sparse,i0:i1:sparse,::sparse,:]
    elif coord == 3:
        pos1 = grid[::sparse,::sparse,i0:i1:sparse,:]
        pos2 = grid[::sparse,::sparse,i0:i1:sparse,:]+flux_t[::sparse,::sparse,i0:i1:sparse,:]

#    if coord == 1:
#        pos1 = grid[:,i0:i1:sparse,::sparse,::sparse]
#        pos2 = grid[:,i0:i1:sparse,::sparse,::sparse]+flux_t[:,i0:i1:sparse,::sparse,::sparse]
#    elif coord == 2:
#        pos1 = grid[:,::sparse,i0:i1:sparse,::sparse]
#        pos2 = grid[:,::sparse,i0:i1:sparse,::sparse]+flux_t[:,::sparse,i0:i1:sparse,::sparse]
#    elif coord == 3:
#        pos1 = grid[:,::sparse,::sparse,i0:i1:sparse]
#        pos2 = grid[:,::sparse,::sparse,i0:i1:sparse]+flux_t[:,::sparse,::sparse,i0:i1:sparse]
    else:
        raise Exception('Invalid coordinate %d'%coord)


    p0,p1 = list(),list()
#    print(pos1.shape,pos1.shape[coord],coord)
    for index in range(pos1.shape[coord-1]):
        if coord==1:
            p0.append(pos1[index]/Angstrom2Bohr) #fluxes in au? 
            p1.append(pos2[index]/Angstrom2Bohr)
        if coord==2:
            p0.append(pos1[:,index]/Angstrom2Bohr) #fluxes in au? 
            p1.append(pos2[:,index]/Angstrom2Bohr)
        if coord==3:
            p0.append(pos1[:,:,index]/Angstrom2Bohr) #fluxes in au? 
            p1.append(pos2[:,:,index]/Angstrom2Bohr)
    p0,p1 = np.array(p0),np.array(p1)

    ind  = np.linalg.norm(p0-p1,axis=-1) >= cutoff #preselection
    obj0 = pymol.PymolObject(p0[ind],
                             p1[ind],
                             name='arrows',
                             type='modevectors',
                             #headrgb=(0.0,0.0,0.0),tailrgb=(0.0,0.0,0.0),
                             headrgb=rgb,tailrgb=rgb,
                             notail=True,
                             head=head,
                             head_length=head_length,
                             cutoff = cutoff,
                             factor = factor) 
    obj = obj0#+obj1+obj2
    obj.write(fn_out)
#    pymol.PymolArrows(fn_out, p1, p2, cutoff=cutoff, factor=factor, timestep=1, name="arrows")

if(__name__ == "__main__"):
    main()

