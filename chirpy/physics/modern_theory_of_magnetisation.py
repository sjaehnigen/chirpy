# -------------------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#    https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
#
#  Copyright (c) 2010-2021, The ChirPy Developers.
#
#
#  Released under the GNU General Public Licence, v3 or later
#
#   ChirPy is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation, either version 3 of the License,
#   or any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.
#   If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------


import numpy as np
from ..physics import constants

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = +1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

# ToDo: clean up


def calculate_mic(e0, r1_pert, sw_c, n_states, r_wc_aa, box_vec_aa):
    sw_m_ic_r = calculate_mic_rotation(e0, r1_pert, n_states, r_wc_aa, box_vec_aa)
    sw_m_ic_t = calculate_mic_translation(sw_c, n_states, r_wc_aa, box_vec_aa)

    return sw_m_ic_r.sum(axis=0).sum(axis=0), sw_m_ic_t.sum(axis=0)


def calculate_mic_rotation(e0, r1_pert, n_states, r_wc_aa, box_vec_aa):
    # case 0 if states interact within box (M_LC), +1 if state in column is
    # considered +R with respect to state in line, et VV.
    case = np.abs(r_wc_aa[:, None, :] - r_wc_aa[:, :])
    case = (case/box_vec_aa[None, None, :]*2).astype(np.int)
    # case [case  < 0] = 0

    v = r1_pert*e0[:, :, None]
    # v  = 2*(data['R1_pert'] - np.swapaxes(data['R1_pert'], 0, 1))
    # * data['E0'][:, :, None]#*abs(case)

    sw_m_ic_r = np.zeros((n_states, n_states, 3))
    # for m in range(3):
    #    for o in range(3):
    #        for p in range(3):
    #            #sw_m_ic_r[:, :, m] += 2*eps[m, p, o]*v[:, :, o]*box_vec_aa[p]*case[:, :, p]*constants.l_aa2au
    #            sw_m_ic_r[:, :, m] += 2*eijk[m, p, o]*v[:, :, o]*box_vec_aa[p]*case[:, :, p]*constants.l_aa2au
    sw_m_ic_r = -2 * np.sum(eijk[None, None, :, :, :]
                            * box_vec_aa[None, None, :, None, None]
                            * case[:, :, :, None, None]
                            * constants.l_aa2au*v[:, :, None, :, None],
                            axis=(2, 3, 4))

    return sw_m_ic_r / 2  # DEBUG sign; I think the case has to be mirrored


def calculate_mic_translation(sw_p_pert, n_states, r_wc_aa, box_vec_aa):
    sw_m_ic_t = np.zeros((n_states, 3))
    t = sw_p_pert
#    for m in range(3):
#        for o in range(3):
#            for p in range(3):
#                #sw_m_ic_t[:, m] += 2*eps[m, p, o]*t[:, o]*box_vec_aa[p]*constants.l_aa2au
#                sw_m_ic_t[:, m] += 2*eijk[m, p, o]*t[:, o]*box_vec_aa[p]*constants.l_aa2au
    # sw_m_ic_t = 2*np.sum(eijk[None, :, :, :]*box_vec_aa[None, :, None, None]*constants.l_aa2au*t[:, None, :, None], axis=(1, 2, 3))
    sw_m_ic_t = -2 * np.sum(eijk[None, :, :, :]
                            * box_vec_aa[None, None, :, None]
                            * constants.l_aa2au
                            * t[:, :, None, None],
                            axis=(1, 2, 3))

#        sw_m_ic_t   = sw_m_ic_t #- TranslateMagneticMoments(data['SW_P_VF_pert'], r_wc_aa)#-data['nuc_coc_au'][None, :]*constants.l_au2aa)    
    return sw_m_ic_t  # /2
