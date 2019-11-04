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

import numpy as np

def read_ener_file(fn):
    with open(fn, 'r') as f:
        title = f.readline()[1:]   # Could be used later to create a data dict of all lists
        steps = f.readlines()

    kin    = list()
    pot    = list()
    temp   = list()
    cqty   = list()
    time   = list()
    step_n = list()

    for step in steps:
        buff = step.split()
        step_n.append(float(buff[0]))
        time.append(float(buff[1]))
        kin.append(float(buff[2]))
        temp.append(float(buff[3]))
        pot.append(float(buff[4]))
        cqty.append(float(buff[5]))

    step_n = np.array(step_n)
    time   = np.array(time)
    kin    = np.array(kin)
    temp   = np.array(temp)
    pot    = np.array(pot)
    cqty   = np.array(cqty)

#    kin_avg = np.average(kin)
#    pot_avg = np.average(pot)
#    cqty_avg= np.average(cqty)

    return step_n,time,temp,kin,pot,cqty


def read_tot_dipole_file(fn):
    """returns total dipole moments in a.u."""

    dat = np.genfromtxt(fn,
                        dtype=None,
                        comments='#',
                        usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                        deletechars='_',
                        autostrip=True
                        )
    dip_au = dat[:, :3]
    dip_de = dat[:, 3:6]
    ddip_au = dat[:, 6:]

    return dip_au, dip_de, ddip_au