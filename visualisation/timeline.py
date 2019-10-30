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

import matplotlib.pyplot as plt
from pylab import polyfit
# needs cleanup


def show_and_interpolate_array(x, y, title, xlabel, ylabel, plot):
    m = list()
    b = list()
    if plot == 1:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
    k, l = polyfit(x, y, 1)
    print("%s trend:\r\t\t\t\t%g*x + %f" % (title, k, l))
    m.append(k)
    b.append(l)
    if plot == 1:
        plt.plot(x, m[0]*x+b[0])
        for i in range(int(len(x)/1000)):
            k, l = polyfit(
                    x[-1000*(i+1):-1000*i-1],
                    y[-1000*(i+1):-1000*i-1],
                    1
                    )
            m.append(k)
            b.append(l)
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


#    plt.savefig(title+'.png')
# ToDo: see below
# def PlotAndShowArray_MultiDim(data,title,xlabel,ylabel,plot):
#     '''data with shape (x,n_yval)'''
#     print(data.shape)
#     if plot == 1:
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         for j in range(1,data.shape[1]): 
#             x = data[:,0]
#             y = data[:,j]
#             color = rcParams['axes.color_cycle'][j-1]  #explicit formulation of pyplot iterlist
#             plt.plot(x,y,':',label=title+str(j-1),c=color)
#     #plt.gca().set_color_cycle(None)
#     for j in range(1,data.shape[1]):
#         color = rcParams['axes.color_cycle'][j-1]
#         m = list()
#         b = list()
#         x = data[:,0]
#         y = data[:,j]
#         k,l = polyfit(x, y, 1)
#         print("%s trend:\r\t\t\t\t%g*x + %g" % (title+str(j-1),k,l))
#         m.append(k)
#         b.append(l)
#         if plot == 1:
#             plt.plot(x,m[0]*x+b[0],"--",c=color)
#             for i in range(int(len(x)/1000)):
#                 k,l = polyfit(x[-1000*(i+1):-1000*i-1], y[-1000*(i+1):-1000*i-1], 1)
#                 m.append(k)
#                 b.append(l)
#                 plt.plot(x[-1000*(i+1):-1000*i-1],m[i+1]*x[-1000*(i+1):-1000*i-1]+b[i+1],'-',c=color)
# 
#     if plot == 1:
#         plt.legend(loc='best')
#         plt.show()

#EOF
