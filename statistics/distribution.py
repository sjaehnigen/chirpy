#!/usr/bin/env python

import numpy as np
from ..mathematics.algebra import rotation_matrix


def radial_distribution_function(DS, DO, cell_au = None, **kwargs):
    '''DS/O ... source/origin data'''
    n_frames, n_O, three = DO.shape
    h_v = kwargs.get('half_vector',None) #alpha version: scan only half the sphere (all vecs before plane)
    del kwargs['half_vector']
   
    def get_P(s,o,_hv=None):
        _P = s - o[:,None]
        if not cell_au is None:  _P -= np.around(_P/cell_au) * cell_au
        if not _hv is None: #beta
            ind = np.array( 
                [ 
                    np.tensordot( 
                        rotation_matrix( 
                            _v, 
                            np.array( [ 0.0, 0.0, 1.0 ] ) 
                        ), 
                        _p, 
                        axes=( [ 1 ], [ 1 ] ) 
                    ).swapaxes( 0,1 ) for _v, _p in zip( _hv, _P) 
            ] )
            _P = _P [ ind[ :, :, 2 ] > 0 ] # returns flattened first 2 dims (which is fine)
        return np.linalg.norm(_P,axis=-1)#.flatten() #auto-flattening?

    _wg = n_O * DS.shape[0] * DS.shape[1] / np.prod(cell_au) # norm to n_frames and density

    if not h_v is None: return np.sum([ _rdf( get_P( DS, DO[:,_o], _hv = h_v[:, _o ] ), **kwargs) for _o in range( n_O ) ], axis=0) / _wg * 2

    else: return np.sum([ _rdf( get_P( DS, DO[:,_o] ), **kwargs) for _o in range( n_O ) ], axis=0) / _wg


def rdf(*args,**kwargs):
    return radial_distribution_function(*args,**kwargs)

# various kernels
def _rdf(_P, rng = ( 0.1, 10 ), bins=100): #pair
    '''pos of shape (n_frames,n_particles,3); ref integer of reference particle'''

    R = np.linspace( *rng, bins )
    rdf = np.histogram( _P, bins = bins, density = False, range = rng )[ 0 ].astype( float )
    rdf /= 4. * np.pi * R**2 * ( R[1] - R[0] ) #divide by shell volume

    return rdf #R return also raxis?

