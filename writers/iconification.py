#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------


import numpy as np
import copy

from scipy.interpolate import UnivariateSpline
#old lib
#from fileio.xyz import WriteXYZFile
Angstrom2Bohr = 1.8897261247828971


#NEEDS Routine to include arrow tip in total vector length

def VMDTube(fn, pos_aa,**kwargs):
    '''pos_au with shape n_points p. tube, n_tubes, 3'''
    arr   = kwargs.get('arrow',False)
    label = kwargs.get('label',False)
    res   = kwargs.get('resolution',10)
    rad   = kwargs.get('radius',0.025)
    sparse= kwargs.get('sparse',5) #sparsity of iconification (NOT of streamline production)
    factor= kwargs.get('factor',1)
    cut   = kwargs.get('cutoff_aa',0) #min length of tube in AA to be iconified
    mat   = kwargs.get('material','AOShiny')
    rhlng =  kwargs.get('relative_head_length',15) #rhlng times radius ... chnage it soon
    rhrad =  kwargs.get('relative_head_radius',3) #rhrad times radius ... chnage it soon

    n_points,n_tubes,three = pos_aa.shape
    def spline(points):
        x = np.arange(points.shape[0])
        spl0 = UnivariateSpline(x,points[:,0])
        spl1 = UnivariateSpline(x,points[:,1])
        spl2 = UnivariateSpline(x,points[:,2])
        return np.array([spl0(x),spl1(x),spl2(x)]).swapaxes(0,1)

    arr_head_sense = lambda p,depth: ((p[-1,None]-p[-depth:-1])/np.linalg.norm(p[-1,None]-p[-depth:-1],axis=-1)[:,:,None]).sum(axis=0)/depth

    ind = np.linalg.norm(np.abs(np.diff(pos_aa[:,:,:3],axis=0)).sum(axis=0),axis=-1) >= cut #consider curve
    #ind = np.linalg.norm(np.abs(pos_aa[-1,:,:3]-pos_aa[0,:,:3]),axis=-1) >= cut #simple
    pos_aa = np.array([spline(points) for points in pos_aa[:,ind,:].swapaxes(0,1)]).swapaxes(0,1)[::sparse][::factor]
    print(pos_aa.shape)
    with open(fn,'w') as f:
        f.write("color change rgb tan 0.0 0.4 0.0\n")
        f.write("draw color tan\n")
        #f.write("draw color black\n")
        #f.write("draw materials off\n")
        f.write("draw materials on\n")
        f.write("draw material %s\n"%mat)
        for ip,p1 in enumerate(pos_aa):
            if ip>0: [f.write("draw cylinder {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} radius %f resolution %d\n"%(
                              t0[0],t0[1],t0[2],
                              t0[0]+1.0*(t1[0]-t0[0]),
                              t0[1]+1.0*(t1[1]-t0[1]),
                              t0[2]+1.0*(t1[2]-t0[2]),
                              rad,res)) for t0,t1 in zip(p0,p1)] 
#            if ip>0: [f.write("draw line {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} width %d style solid\n"%(t0[0],t0[1],t0[2],t1[0],t1[1],t1[2],2)) for t0,t1 in zip(p0,p1)] 
            p0=copy.deepcopy(p1)
            #[f.write("draw sphere {%8.2f %8.2f %8.2f} radius %f resolution %d\n"%(t1[0],t1[1],t1[2],rad,res)) for t1 in p1] 
        if arr:
            [f.write("draw cone {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} radius %f resolution %d\n"%(
                     t0[0],t0[1],t0[2],
                     t0[0]+rhlng*rad*t1[0],
                     t0[1]+rhlng*rad*t1[1],
                     t0[2]+rhlng*rad*t1[2],
                     rhrad*rad,rhrad*res)) for t0,t1 in zip(pos_aa[-1],arr_head_sense(pos_aa,5))]


def VMDLine(fn, pos_aa,**kwargs):
    '''pos_au with shape n_points p. tube, n_tubes, 3'''
    arr   = kwargs.get('arrow',False)
    label = kwargs.get('label',False)
    res   = kwargs.get('resolution',10)
    sty   = kwargs.get('style','solid')
    wdt   = kwargs.get('width',1)
    rad   = kwargs.get('radius',0.025) #only for arrow, NEXT change routine
    sparse= kwargs.get('sparse',5) #sparsity of iconification (NOT of streamline production)
    factor= kwargs.get('factor',1)
    cut   = kwargs.get('cutoff_aa',0) #min length of tube in AA to be iconified

    n_points,n_tubes,three = pos_aa.shape
    def spline(points):
        x = np.arange(points.shape[0])
        spl0 = UnivariateSpline(x,points[:,0])
        spl1 = UnivariateSpline(x,points[:,1])
        spl2 = UnivariateSpline(x,points[:,2])
        return np.array([spl0(x),spl1(x),spl2(x)]).swapaxes(0,1)

    arr_head_sense = lambda p,depth: ((p[-1,None]-p[-depth:-1])/np.linalg.norm(p[-1,None]-p[-depth:-1],axis=-1)[:,:,None]).sum(axis=0)/depth

    ind = np.linalg.norm(np.abs(np.diff(pos_aa[:,:,:3],axis=0)).sum(axis=0),axis=-1) >= cut #consider curve
    #ind = np.linalg.norm(np.abs(pos_aa[-1,:,:3]-pos_aa[0,:,:3]),axis=-1) >= cut #simple
    pos_aa = np.array([spline(points) for points in pos_aa[:,ind,:].swapaxes(0,1)]).swapaxes(0,1)[::sparse][::factor]
    print(pos_aa.shape)
    with open(fn,'w') as f:
        #f.write("color change rgb tan 0.0 0.4 0.0\n")
        #f.write("draw color tan\n")
        f.write("draw color black\n")
        #f.write("draw materials off\n")
        f.write("draw materials on\n")
#        f.write("draw material AOShiny\n")
        for ip,p1 in enumerate(pos_aa):
            if ip>0: [f.write("draw line {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} width %d style %s\n"%( #add style
                              t0[0],t0[1],t0[2],
                              t0[0]+1.8*(t1[0]-t0[0]),
                              t0[1]+1.8*(t1[1]-t0[1]),
                              t0[2]+1.8*(t1[2]-t0[2]),
                              wdt,sty)) for t0,t1 in zip(p0,p1)] 
#            if ip>0: [f.write("draw line {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} width %d style solid\n"%(t0[0],t0[1],t0[2],t1[0],t1[1],t1[2],2)) for t0,t1 in zip(p0,p1)] 
            p0=copy.deepcopy(p1)
            #[f.write("draw sphere {%8.2f %8.2f %8.2f} radius %f resolution %d\n"%(t1[0],t1[1],t1[2],rad,res)) for t1 in p1] 
        if arr:
            [f.write("draw cone {%16.9f %16.9f %16.9f} {%16.9f %16.9f %16.9f} radius %f resolution %d\n"%(
                     t0[0],t0[1],t0[2],
                     t0[0]+15*rad*t1[0],
                     t0[1]+15*rad*t1[1],
                     t0[2]+15*rad*t1[2],
                     3*rad,3*res)) for t0,t1 in zip(pos_aa[-1],arr_head_sense(pos_aa,5))]

#draw color black
#draw materials off
#draw cylinder {6 6 4} {6 6 4.2} radius 7.2 resolution 50
#color change rgb tan 0.0 0.4 0.0
#draw color tan
#draw materials off
#draw cylinder {6 6 4} {6 6 10} radius 0.1 resolution 50
#draw cone {6 6 10} {6 6 11} radius 0.3 resolution 50
#draw materials off
#draw text {5.3 6 11.2} "r" size 2 thickness 5
#draw text {5.0 6 11.5} "z" size 1 thickness 3
#draw color orange
#draw materials off
#draw cylinder {6 6 4} {10.243 10.243 4} radius 0.1 resolution 50
#draw cone {10.243 10.243 4} {10.950 10.950 4} radius 0.3 resolution 50
#draw materials off
#draw text {11.332 11.232 3.0} "r" size 2 thickness 5
#draw text {11.032 11.232 3.3} "xy" size 1 thickness 3

