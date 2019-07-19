#!/usr/bin/env python

import numpy as np
from numpy.fft import fftfreq, ifftn, fftn

def k_get_cell(n1, n2, n3, a1, a2, a3):            
    r1 = np.arange(n1)*(a1/n1)-a1/2
    r2 = np.arange(n2)*(a2/n2)-a2/2
    r3 = np.arange(n3)*(a3/n3)-a3/2
    k1 = 2*np.pi*fftfreq(n1,a1/n1)
    k2 = 2*np.pi*fftfreq(n2,a2/n2)
    k3 = 2*np.pi*fftfreq(n3,a3/n3)
    ix, iy, iz = (slice(None), None, None), (None, slice(None), None), (None, None, slice(None))
    (X, Kx) = (r1[ix], k1[ix])
    (Y, Ky) = (r2[iy], k2[iy])
    (Z, Kz) = (r3[iz], k3[iz])
    R = np.sqrt(X**2 + Y**2 + Z**2)
    K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    return R,K

def _k_v1(k):
    """Fourier transform of Coulomb potential $1/r$"""
    with np.errstate(divide='ignore'):
        return np.where(k==0.0, 0.0, np.divide(4.0*np.pi, k**2))
            
def k_potential(data, cell_au):
    n1, n2, n3 = data.shape
    a1, a2, a3 = tuple(cell_au.diagonal())
    R,K        = k_get_cell(n1, n2, n3, a1*n1, a2*n2, a3*n3)
    V_R        = ifftn(_k_v1(K)*fftn(data)).real
    return R, V_R

#STILL UNCLEAR
#def calculate_G0(j, r):
#    #Magnetic permittivity of the vacuum
#    B = np.cross(r,j,axisa=0,axisb=0,axisc=0)
#    B = B.sum(axis=(1, 2, 3)) / 3
#    return B

