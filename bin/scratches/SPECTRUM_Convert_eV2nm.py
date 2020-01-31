#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------


import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f' , '--fn_in',          default=None)
    parser.add_argument('-o',  '--fn_out',         default='out.dat')
    parser.add_argument('--no_header', action='store_true', default=False)
    #Later: give starting and ending unit
    args = parser.parse_args()

    # S.I. constants
    _c = 2.99792458E+08 # speed of light [m/s]
    _h = 6.626070040E-34 # Planck's constant [Js]
    #hbar_si = h_si/(2*pi) # reduced planck constant [Js]
    #m_p_si = 1.672621898E-27 # mass protron [kg]
    #m_e_si = 9.10938356E-31 # mass electron [kg]
    #m_amu_si = 1.660539040E-27 # atomic mass unit [kg]
    _e = 1.6021766208E-19 # unit charge [C]
    #eps0_si = 8.854187817E-12 # vacuum permittivity
    #a0_si = 0.52917721067E-10 # Bohr radius [m]
    #k_B_si = 1.38064852E-23 # Boltzmann constant [J/K]

    #to J
    _eV2J = lambda x: _e * x
    # _nm2J = lambda x: _h * _c / x * 1.E-9 
    # inhomogeneous sparsity of points => could be interpolated (or ignored)

    #from J
    # _J2eV = lambda x: 1 / _e * x
    _J2nm = lambda x: _h * _c / x * 1.E+9

    _in  = _eV2J
    _out = _J2nm
    data = np.loadtxt(args.fn_in)[1-int(args.no_header):].astype(float)
    print(data.shape)
    with open(args.fn_out,'w') as f: f.write(''.join(["%12.6f %12.6f\n"%p for p in zip(map(_out,map(_in,data[:,0])),data[:,1])]))

if __name__ == "__main__":
    main()
