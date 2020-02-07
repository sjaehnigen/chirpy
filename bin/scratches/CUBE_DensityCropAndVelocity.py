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


import copy
from chirpy.classes import quantum

rho_cube = 'DENSITY-000001.cube'
fn = rho_cube
j_cube = 'CURRENT-000001-%d.cube'
fn1 = j_cube%1
fn2 = j_cube%2
fn3 = j_cube%3

system = quantum.ElectronicSystem(fn,fn1,fn2,fn3)

system.auto_crop(thresh=system.rho.threshold/2)
#system.calculate_velocity_field(lower_thresh=5.E-2) #default thresh=1.E-3

#out = copy.deepcopy(system.v)
#fn_cube = 'VELOCITY-000001-%d-CROPPED.cube'
#fn1 = fn_cube%1
#fn2 = fn_cube%2
#fn3 = fn_cube%3
#out.write(fn1,fn2,fn3)
#del out

out = copy.deepcopy(system.rho)
fn_cube = ''.join(rho_cube.split('.')[:-1])+'-CROPPED.'+rho_cube.split('.')[-1]
fn = fn_cube
out.write(fn)
del out

#out = copy.deepcopy(system.j)
#fn_cube = ''.join(j_cube.split('.')[:-1])+'-CROPPED.'+j_cube.split('.')[-1]
#fn1 = fn_cube%1
#fn2 = fn_cube%2
#fn3 = fn_cube%3
#out.write(fn1,fn2,fn3)
#del out


#system.rho.aim()
#for i in range(system.rho.n_atoms):
#    fn_cube = 'atom-%d.cube'%i
#    system.rho.aim_atoms[i].write(fn_cube,comment2='AIM atom %d'%i)

#PYMOL_vels2arrows.py GEOMETRY.xyz --cutoff 0.05 --scale 2000 --f vectors.pymolrc --head_length 0.3 --head 0.05  --tail 0.015 --color_rgb 0 0 0 --factor 0.2
#PYMOL_vectorfield2arrows.py VELOCITY-000001-{1..3}.cube --cutoff 0.05 --scale 2000 --f velocity_00.pymolrc --head 0.05 --sparse 5 --color_rgb 1 0 1 --factor 0.2

