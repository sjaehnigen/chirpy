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

import matplotlib
import matplotlib.pyplot as plt
from pylab import polyfit

matplotlib.use('TkAgg')


def show_and_interpolate_array(x, y, title, xlabel, ylabel, plot):
    m = list()
    b = list()
    if plot == 1:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
    _m, _b = polyfit(x, y, 1)
    print("%s trend:\r\t\t\t\t%g*x + %f" % (title, _m, _b))
    m.append(_m)
    b.append(_b)
    if plot == 1:
        plt.plot(x, m[0]*x+b[0])
        for i in range(int(len(x)/1000)):
            _m, _b = polyfit(
                    x[-1000*(i+1):-1000*i-1],
                    y[-1000*(i+1):-1000*i-1],
                    1
                    )
            m.append(_m)
            b.append(_b)
            plt.plot(
                    x[-1000*(i+1):-1000*i-1],
                    m[i+1]*x[-1000*(i+1):-1000*i-1]+b[i+1]
                    )
#    if len(x) >= 1000:
#        m2,b2 = polyfit(x[-1000:], y[-1000:], 1)
#        plt.plot(x[-1000:],m2*x[-1000:]+b2)
#    if len(x) >= 2000:
#        m3,b3 = polyfit(x[-2000:-1000], y[-2000:-1000], 1)
#        plt.plot(x[-2000:-1000],m3*x[-2000:-1000]+b3)
#    if len(x) >= 3000:
#        m4,b4 = polyfit(x[-3000:-2000], y[-3000:-2000], 1)
#        plt.plot(x[-3000:-2000:],m4*x[-3000:-2000:]+b4)
    if plot == 1:
        plt.show()
