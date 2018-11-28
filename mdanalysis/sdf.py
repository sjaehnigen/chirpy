#!/usr/bin/env python

import MDAnalysis as mda
import sys,copy

import numpy as np
from collections import OrderedDict
from lib import constants
from tools.algebra import RotationMatrix
from analysis.pcbtools import InitialiseTopology_PDB
from analysis.utils import AlignTrajectory

def InitialiseGrid(u,mesh,size,origin=None): 
    grid=list()
    if origin==None:
        origin=np.zeros((3))
        for z,i in enumerate(u.dimensions[0:3]):#Origin lies in the center of the box
            origin[z]=i/2-size
    for i in origin:
        grid.append(np.arange(i,i+2*size+mesh,mesh))
    grid = np.array(grid)
    return grid,origin

def KabschAlgorithm(P,ref):
    C = np.dot(np.transpose(ref), P)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
    return np.dot(V, W) 

def CalculateSDF_PCB(u,minf,maxf,step,residue,cutoff,selections,mesh,update_sel=True,ext_dims=False):
    if residue=='PCB':
        map  = InitialiseTopology_PDB(u)
    u.trajectory[0]
    if ext_dims.all() != False:
        u.dimensions = ext_dims
    grid,origin = InitialiseGrid(u,mesh,20) #This could also be a customised option
    data = OrderedDict()
    track= OrderedDict()
    resids=list()
    sel_all = '('
    for sel in selections:
        data[sel]  = np.zeros((grid[0].shape[0],grid[1].shape[0],grid[2].shape[0]))
        track[sel] = list()
        sel_all    += sel + ') or ('
    sel_all    = sel_all[:-5]
    res        = u.select_atoms('resname %s'%residue)
    ind_res    = res.indices
    pos_aa_res = list()

    if update_sel == False:
        print('You switched off updating selection every frame.')
        dims = u.dimensions[0:3]
        cog = res.center_of_geometry()
        u.atoms.translate(0.5*dims-cog)
        u.atoms.wrap()
        if cutoff != 0:
            pre_sel = u.select_atoms(sel_all + ' and ' + '(around %f (resname %s))'%(cutoff,residue))
        elif cutoff==0:
            pre_sel = u.select_atoms(sel_all + ' and ' + 'resname %s'%(residue))

    count=0
    for ts in range(minf,maxf,step):    
        count+=1
        print('\r%6s/%6s (%5.2f %%)'%(ts,maxf,float(100*(ts-minf))/float((maxf-minf)))),
        sys.stdout.flush()
        if residue == 'PCB':
            try:
                map = InitialiseTopology_PDB(u)[0] #PCB dependent
            except:
                print('ERROR: Could not initialise PCB topology. Is there PCB at all?')
                sys.exit(1)
    #Center residue and wrap (accessing u is time consuming, the numpy stuff is fast) 
        u.trajectory[ts]
        if ext_dims.all() != False:
            u.dimensions = ext_dims
        dims = u.dimensions[0:3]
        #cog = u.select_atoms('resname %s'%residue).center_of_geometry()
        cog = res.center_of_geometry()
        u.atoms.translate(0.5*dims-cog)
        u.atoms.wrap()    
        pos_aa = u.atoms.coordinates()
#FURTHER>>>This is the only PCB-dependent part that needs to be generalised
    #Define plane and rotate everything
        if residue == 'PCB':
            vec1 = pos_aa[int(map[5 ]),:] - pos_aa[int(map[11]),:] 
            vec2 = pos_aa[int(map[17]),:] - pos_aa[int(map[11]),:] 
            normal = np.cross(vec1,vec2)
            normal /= np.linalg.norm(normal)
            R1 = RotationMatrix(normal,np.array([0.0,0.0,1.0]))    
            a1 = np.asarray(np.dot(R1,vec1)).squeeze()
            b1 = np.asarray(np.dot(R1,vec2)).squeeze()
            R2 = RotationMatrix(a1+b1,np.array([0.0,-1.0,0.0]))    
            #a  = np.asarray(np.dot(R2,a1)).squeeze()
            #b  = np.asarray(np.dot(R2,b1)).squeeze()    
            R  = np.tensordot(R2,R1,axes=([1],[0]))
            
            pos_aa_rot = pos_aa - 0.5*dims[np.newaxis,:]
            pos_aa_rot = np.tensordot(R,pos_aa_rot,axes=([1],[1])).swapaxes(0,1)
            pos_aa_rot += 0.5*dims[np.newaxis,:]
##<<<
        else:
            pos_aa_rot = pos_aa - 0.5*dims[np.newaxis,:]
            if ts==0:
                pos_aa_ref = pos_aa_rot.copy()
            U = KabschAlgorithm(pos_aa_rot[ind_res],pos_aa_ref[ind_res])
            pos_aa_rot = np.tensordot(U,pos_aa_rot,axes=([1],[1])).swapaxes(0,1)
            pos_aa_rot += 0.5*dims[np.newaxis,:]
    
        #vec1 = pos_aa_rot[int(map[5 ]),:] - pos_aa_rot[int(map[11]),:] 
        #print(a1,vec1)    
        
    #Select residue environment (update selection each frame)
        if update_sel == True:
	        if cutoff != 0:
                    pre_sel = u.select_atoms(sel_all + ' and ' + '(around %f (resname %s))'%(cutoff,residue))
                elif cutoff == 0:
                    pre_sel = u.select_atoms(sel_all + ' and ' + 'resname %s'%(residue))
        resids.append([ts,pre_sel.resids,pre_sel.resnames,pre_sel.segids])
        for sel in selections:
            try:
                res_env = pre_sel.select_atoms(sel).indices
                pos_aa_rot_env = pos_aa_rot[res_env,:]
                x = pos_aa_rot_env[:,0]
                y = pos_aa_rot_env[:,1]
                z = pos_aa_rot_env[:,2]
                idx = (np.abs(grid[0][np.newaxis,:]-x[:,np.newaxis])).argmin(axis=1)
                idy = (np.abs(grid[1][np.newaxis,:]-y[:,np.newaxis])).argmin(axis=1)
                idz = (np.abs(grid[2][np.newaxis,:]-z[:,np.newaxis])).argmin(axis=1)    
                fx = np.abs(x-grid[0][idx])
                fy = np.abs(y-grid[1][idy])
                fz = np.abs(z-grid[2][idz])
                dx = (fx/(x-grid[0][idx])).astype(int)
                dy = (fy/(y-grid[1][idy])).astype(int)
                dz = (fz/(z-grid[2][idz])).astype(int)
                hx = 1-fx/mesh
                hy = 1-fy/mesh
                hz = 1-fz/mesh
                here  = np.sqrt(hx**2+hy**2+hz**2)
                there = 1-here
                data[sel][idx,idy,idz]          += here
                data[sel][idx+dx,idy+dy,idz+dz] += there
                track[sel].append(ts)
            except IndexError:
                print('WARNING: Zero-atom selection: "%s" from "%s" does not give any match!'%(sel,sel_all))
                pass
            except: #NoDataError is not defined ??
                print('WARNING: There is nothing in %s'%sel_all)
                pass
        pos_aa_res.append(pos_aa_rot[ind_res,:])
    
    print('\r%6s/%6s (%5.2f %%)'%(maxf,maxf,float(100)))
    for sel in selections:
#        print(data[sel].shape)
        voxel = np.prod([1/float(v) for v in data[sel].shape])
#        print(voxel)
        data[sel] /= voxel #per unit volume -> true density
        data[sel] /= count #per frame -> occupation in percent

    #pos_aa_res  = np.mean(np.array(pos_aa_res),axis=0) #Averaged structure
    pos_aa_res  = np.array(pos_aa_res)[0] 
    symbols_res = [name[0] for name in u.select_atoms('resname %s'%residue).names]

    return data,track,resids,pos_aa_res,symbols_res,origin

#def CalculateSDF_GeneralSelections():
#EOF
