#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------

import copy
import numpy as np

from ..topology.dissection import fermi_cutoff_function
from .classical_electrodynamics import magnetic_dipole_shift_origin
from .statistical_mechanics import spectral_density
from ..physics import constants
# from ..physics.constants import eijk


def get_power_spectrum(val, **kwargs):
    '''Expects data of shape (n_frames, n_atoms, three)
       No support of trajectory iterators.
       The output is averaged over the no. of atoms/species.
       Optionally returns time-correlation function (return_tcf=True).
       '''
    n_frames, n_atoms, three = val.shape
    wgh = kwargs.get('weights', np.ones(n_atoms))
    r_tcf = kwargs.get('return_tcf', False)
    wgh = np.array(wgh)
    omega, S, R = zip(*[spectral_density(_v, **kwargs)
                        for _v in val.swapaxes(0, 1)])

    data = {}
    # ToDo: add units to variable names (GHz)
    data['power'] = (np.array(S) * wgh[:, None]).sum(axis=0) / n_atoms
    data['omega'] = omega[0]
    if r_tcf:
        data['tcf_velocities'] = np.array(R).sum(axis=0) / n_atoms

    return data

# ToDo: implement TCF return and data dict, average over atoms (no because of summed moment)?

def get_ira_and_vcd(c, m, pos_au, **kwargs):
    origins_au = kwargs.get('origins_au', np.array([[0.0, 0.0, 0.0]]))
    cell_au = kwargs.get('cell_au')
    cutoff_aa = kwargs.get('cutoff_aa', 0)
    cut_type = kwargs.get('cut_type', 'soft')

    # BETA: correlate moment of part with total moments, expects one integer
    subparticle = kwargs.get('subparticle')
    # BETA: correlate moment of one part with another part,
    #     expects tuple of integers
    #     notparticles: exclude these indices; expects list of integers
    subparticles = kwargs.get('subparticles')
    subnotparticles = kwargs.get('subnotparticles')
    # BETA: remove direct correlation between background
    cutoff_bg_aa = kwargs.get('background_correction_cutoff_aa')

    spectrum = list()
    for _i, _o in enumerate(origins_au):
        _c = copy.deepcopy(c)
        _m = copy.deepcopy(m)

        # R_I(t) - R_J(t) #used in manuscript
        _trans = pos_au-_o[:, None, :]
        if cell_au is not None:
            _trans -= np.around(_trans/cell_au) * cell_au

        # R_I(t) - R_J(0) #gauge invariant according to Rodolphe/Arne
        # _trans = pos_au-_o[0,None,None,:]
        # _trans -= np.around(_trans/cell_au)*cell_au
        # _trans += _o[:,None,:] - _o[0,None,None,:]
        _m += magnetic_dipole_shift_origin(_c, _trans)

        if cut_type == 'soft':  # for larger cutoffs
            _scal = fermi_cutoff_function(
                            np.linalg.norm(_trans, axis=2),
                            cutoff_aa * constants.l_aa2au,
                            0.125 * constants.l_aa2au
                            )
            _c *= _scal[:, :, None]
            _m *= _scal[:, :, None]

        if cut_type == 'hard':  # cutoff <2 aa
            _ind = np.linalg.norm(_trans, axis=2) > cutoff_aa*constants.l_aa2au
            _c[_ind, :] = np.array([0.0, 0.0, 0.0])
            _m[_ind, :] = np.array([0.0, 0.0, 0.0])

        if type(cutoff_bg_aa) is float:
            _c_bg = copy.deepcopy(_c)
            # only direct correlation, no transport term!
            # _m_bg = copy.deepcopy(m)
            # --- complete background
            _m_bg = copy.deepcopy(_m)
            if cut_type == 'soft':
                _m_bg *= _scal[:, :, None]
            if cut_type == 'hard':
                _m[_ind, :] = np.array([0.0, 0.0, 0.0])
            _ind_bg = np.linalg.norm(
                   _trans, axis=2) <= cutoff_bg_aa*constants.l_aa2au
            _c_bg[_ind_bg, :] = np.array([0.0, 0.0, 0.0])
            _m_bg[_ind_bg, :] = np.array([0.0, 0.0, 0.0])

        if all([_k is None for _k in [
                                subparticle,
                                subparticles,
                                subnotparticles
                                ]]):
            _c = _c.sum(axis=1)
            _m = _m.sum(axis=1)
            x_spec, ira = calculate_spectral_density(_c, **kwargs)
            x_spec, vcd = calculate_spectral_density(_c, _m, **kwargs)
            if type(cutoff_bg_aa) is float:
                _c_bg = _c_bg.sum(axis=1)
                _m_bg = _m_bg.sum(axis=1)
                ira -= calculate_spectral_density(_c_bg, **kwargs)[1]
                vcd -= calculate_spectral_density(_c_bg, _m_bg, **kwargs)[1]

        elif type(subparticle) is int:
            _c1 = _c.sum(axis=1)
            _c2 = _c[:, subparticle]
            _m1 = _m.sum(axis=1)
            _m2 = _m[:, subparticle]
            x_spec, ira = calculate_spectral_density(_c1, _c2, **kwargs)
            x_spec, vcd1 = calculate_spectral_density(_c1, _m2, **kwargs)
            x_spec, vcd2 = calculate_spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1+vcd2)/2

            if type(cutoff_bg_aa) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = _c_bg[:, subparticle]
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = _m_bg[:, subparticle]
                ira -= calculate_spectral_density(_c1_bg, _c2_bg, **kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(
                                                _c1_bg, _m2_bg, **kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(
                                                _c2_bg, _m1_bg, **kwargs)[1]

        elif type(subnotparticles) is list:
            _c1 = _c.sum(axis=1)
            _c2 = copy.deepcopy(_c)
            _c2[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
            _c2 = _c2.sum(axis=1)
            _m1 = _m.sum(axis=1)
            _m2 = copy.deepcopy(_m)
            _m2[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
            _m2 = _m2.sum(axis=1)
            x_spec, ira = calculate_spectral_density(_c1, _c2, **kwargs)
            x_spec, vcd1 = calculate_spectral_density(_c1, _m2, **kwargs)
            x_spec, vcd2 = calculate_spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1+vcd2)/2
            if type(cutoff_bg_aa) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = copy.deepcopy(_c_bg)
                _c2_bg[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
                _c2_bg = _c2_bg.sum(axis=1)
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = copy.deepcopy(_m_bg)
                _m2_bg[:, subnotparticles] = np.array([0.0, 0.0, 0.0])
                _m2_bg = _m2_bg.sum(axis=1)
                ira -= calculate_spectral_density(_c1_bg, _c2_bg, **kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(
                                                _c1_bg, _m2_bg, **kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(
                                                _c2_bg, _m1_bg, **kwargs)[1]

        elif type(subparticles) is tuple:
            _c1 = _c[:, subparticles[0]]
            _c2 = _c[:, subparticles[1]]
            _m1 = _m[:, subparticles[0]]
            _m2 = _m[:, subparticles[1]]
            x_spec, ira = calculate_spectral_density(_c1, _c2, **kwargs)
            x_spec, vcd1 = calculate_spectral_density(_c1, _m2, **kwargs)
            x_spec, vcd2 = calculate_spectral_density(_c2, _m1, **kwargs)
            vcd = (vcd1+vcd2)/2
            if type(cutoff_bg_aa) is float:
                raise NotImplementedError(
                                    'Background correction not '
                                    'implemented for subparticles option!'
                                    )

        else:
            raise ValueError('Subparticle(s) is (are) not an integer '
                             '(a tuple of integers)!')

        _cc = constants.current_current_prefactor(300)
        _cm = constants.current_magnetic_prefactor(x_spec, 300)
        spectrum.append([x_spec, ira*_cc, vcd*_cm])
    spectrum = np.array(spectrum).sum(axis=0)/origins_au.shape[0]

    return spectrum
