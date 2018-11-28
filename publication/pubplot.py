#!/usr/bin/env python3

import os
import sys
import numpy as np
import copy
from itertools import cycle
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.patches import ArrowStyle

#Angstrom2Bohr = 1.8897261247828971

def source_params(matplotlib):
    matplotlib.rc('xtick', labelsize=30)
    matplotlib.rc('ytick', labelsize=30)
    matplotlib.rc('font',  size=30)
    
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype' ] = 42 
    #matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) #gives warning
    matplotlib.rcParams['mathtext.fontset'] = 'stixsans' 
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{upgreek}',
                                                  #r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
                                                  #r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts 
                                                  r'\usepackage{helvet}',    # set the normal font here
                                                  r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                                                  r'\sansmath',               # <- tricky! -- gotta actually tell tex to use!]#,r'\usepackage[cm]{sfmath}']
                                                  r'\def\mymathhyphen{{\hbox{-}}}']
    matplotlib.rcParams.update({'mathtext.default':  'regular' })

def make_nice_ax(p): 
    '''p object ... AxesSubplot'''
    p.tick_params('both', length=5, width=3, which='minor')
    p.tick_params('both', length=10, width=3, which='major')
    p.tick_params(axis='both',which='both',pad=10,direction='in',top=False,right=False)
    p.yaxis.set_ticks_position('left')
      
    #p.spines['top'].set_visible(False)
    p.spines['top'].set_linewidth(3.0)    
    p.spines['bottom'].set_linewidth(3.0)
    p.spines['left'].set_linewidth(3.0)
    p.spines['right'].set_linewidth(3.0)


def plot_local_vs_global(ax,x1,x2,l,g,**kwargs):
    fill = kwargs.get('fill',False)
    bool_g = kwargs.get('do_global',True) #These bools are nice for beamer, because spectra are included bit not shown
    bool_l = kwargs.get('do_local',True)
    std_l = kwargs.get('std_l')
    std_g = kwargs.get('std_g')
    _exp = kwargs.get('exp') #,np.array([np.linspace(0,4000,100),np.zeros((100))]).T) #cheap workaround
    if _exp is not None: e,xe = _exp[:,1],_exp[:,0]
    xlim = kwargs.get('xlim',(1800,800))
    ylim = kwargs.get('ylim')
    sep = kwargs.get('sep',5) #separation between plots in percent
    color1 = kwargs.get('color1','mediumblue')
    color2 = kwargs.get('color2','crimson')

    
    if fill and any(_a is None for _a in [std_l,std_g]):
        raise Exception('ERROR: Needs _std arrays for "fill" option!')
    
    ############################# Calculate hspace per plot and ylim #################
    _slc1 = slice(*sorted(np.argmin(np.abs(x1-_x)) for _x in xlim))
    _slc2 = slice(*sorted(np.argmin(np.abs(x2-_x)) for _x in xlim))
    if _exp is not None: _slce = slice(*sorted(np.argmin(np.abs(xe-_x)) for _x in xlim))
    
    if _exp is not None: _shift=(1+sep/100)*max([np.amax(_s)-np.amin(_s) for _s in (l[_slc1],g[_slc2],e[_slce])])
    else: _shift=(1+sep/100)*max([np.amax(_s)-np.amin(_s) for _s in (l[_slc1],g[_slc2])])
    print(_shift)

    _l = l
    _g = g-  _shift
    if _exp is not None: _e = e-2*_shift

    if ylim is None:
        if _exp is not None: ylim = (np.amin(_e)-0.25*_shift,np.amax(_l)+0.25*_shift)
        else: ylim = (np.amin(_g)-0.25*_shift,np.amax(_l)+0.25*_shift)

    ############################# plot reference (experiment) #################
    #x_exp   = exp[:,0]

    #_e = -6+exp[:,1]*3
    #vcd_exp = -8+exp_vcd[:,1]#*2000

    if _exp is not None: ax.plot(xe,_e,'-',lw=3,color='black',label='exp.')
    
    
    ############################# plot local data #############################    
    if bool_l:
        if fill: ax.fill_between(x1, _l+std_l, _l-std_l, color=color1, alpha=.25)
        ax.plot(x1,_l,'-',lw=3,color=color1)
    
    ############################# plot global data ############################
    if bool_g:
        if fill: ax.fill_between(x2, _g+std_g, _g-std_g, color=color2, alpha=.25)
        ax.plot(x2,_g,'-',lw=3,color=color2)

    ############################# layout ######################################
    #for p in ax:
    #    p.set_xlim(*xlim)
    make_nice_ax(ax)
    #    p.set_ylabel(next(ylabel))
    
        #p.xaxis.set_major_locator(MultipleLocator(200))
        #p.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #p.xaxis.set_minor_locator(MultipleLocator(100))  # for the minor ticks, use no labels; default NullFormatter
        #p.yaxis.set_major_locator(MultipleLocator(1))
        #p.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        #p.yaxis.set_minor_locator(MultipleLocator(0.5))    
    
   
    if False:    
        if do_local or do_global: ax[0].text(1400,1.7,r'\emph{calc.}',color='black',size=26)
        if do_global: ax[0].text(1400,-3.5,r'\textbf{w/ solvent}',color=color2,size=26) 
        if do_local:  ax[0].text(1400,-0.5,r'\textbf{local}',color=color1,size=26) 
        if do_local:  ax[0].text(1610,-1.0,r'w/o supramolecular correlations',color=color1,size=20) 
        ax[0].text(1400,-5.2,r'\emph{exp.}',size=26) 

        if do_local or do_global: ax[1].text(1400,1.5,r'\emph{calc.}',color='black',size=26) 
        if do_global: ax[1].text(1400,-5.5,r'\textbf{w/ solvent}',color=color2,size=26) 
        if do_local:  ax[1].text(1400,-1.5,r'\textbf{local}',color=color1,size=26) 
        if do_local:  ax[1].text(1610,-2.2,r'w/o supramolecular correlations',color=color1,size=20) 
        ax[1].text(1400,-7.7,r'\emph{exp.}',size=26) 

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #ax.set_ylim(-6.5,3.1)
    #ax[0].yaxis.set_major_locator(MultipleLocator(2))
    #ax[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    #ax[0].yaxis.set_minor_locator(MultipleLocator(1))    

    #ax[1].set_ylim(-10.4,2.9)
    #ax[1].yaxis.set_major_locator(MultipleLocator(2))
    #ax[1].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    #ax[1].yaxis.set_minor_locator(MultipleLocator(1))    

    a_s=ArrowStyle("-|>", head_length=0.4,head_width=0.1)

    my_arrow_p = dict(arrowstyle=a_s,lw=1,ec='gray',fc='gray',ls='solid',connectionstyle="arc,angleA=-90,angleB=90,armA=50,armB=150,rad=30")
    text_props = dict(color="gray",fontsize=20)

    #my_arrow_p = dict(arrowstyle=a_s,lw=1,ec='gray',fc='gray',ls='solid',connectionstyle="arc,angleA=-90,angleB=90,armA=10,armB=30,rad=10")

    #zz=1/x_scaling
    if False:    
        exp_tags  = [1622, 1590, 1523, 1465, 1420, 1370, 1310, 1245, 1160, 1125, 1045, 1025,  940,  860]
        exp_yshift1=[-4.5, -4.8, -6.3, -6.3, -5.5, -5.3, -5.5, -6.3, -6.3, -6.3, -6.3, -6.3, -6.3, -6.3]
        exp_yshift2=[-6.1, -8.6, -8.2, -8.0, -6.2, -8.4, -6.4, -6.7, -7.8, -7.9, -6.8, -7.9, -7.8, -7.8]
        cal_tags  = [zz*1565,zz*1530,zz*1480,zz*1405,zz*1380,zz*1335,zz*1290,zz*1240,zz*1155,zz*1105,zz*1025,zz* 990,zz* 895,zz* 830]
        cal_yshift1=[-3.0, -3.0, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6, -3.6]
        cal_yshift2=[-0.8, -4.8, -3.5, -3.3, -2.0, -3.9, -1.8, -2.2, -3.3, -3.4, -2.0, -3.4, -3.3, -3.3]    

        for itag,tag in enumerate(zip(exp_tags,exp_yshift1,exp_yshift2,cal_tags,cal_yshift1,cal_yshift2)):
            ax[1].annotate(r"\textbf{%d}"%(itag+1), xy=(tag[0], tag[2]),**text_props)
            ax[1].annotate(r"\textbf{%d}"%(itag+1), xy=(tag[3], tag[5]),**text_props)
        
            #ax[0].annotate("", xy=(tag[0], tag[1]), xytext=(tag[3], tag[4]),arrowprops=my_arrow_p)
            #ax[1].annotate("", xy=(tag[0], tag[2]), xytext=(tag[3], tag[5]),arrowprops=my_arrow_p)

#TODO: GENERALISE THIS FUNCTION TO "PLOT_MULTIPLE"
def multiplot(ax,x_a,y_a,**kwargs):
    fill = kwargs.get('fill',False)
    bool_a = kwargs.get('bool_a') #list of bools which spectra to unhide
    std_a = kwargs.get('std_a')
    _exp = kwargs.get('exp') #,np.array([np.linspace(0,4000,100),np.zeros((100))]).T) #cheap workaround
    if _exp is not None: e,xe = _exp[:,1],_exp[:,0]
    xlim = kwargs.get('xlim',(1800,800))
    ylim = kwargs.get('ylim')
    sep = kwargs.get('sep',5) #separation between plots in percent
    color_a = kwargs.get('color_a',cycle(['mediumblue','crimson','green','goldenrod'])) #list of colors
    sty_a = kwargs.get('style_a')
    alpha_a = kwargs.get('alpha_a')
    stack = kwargs.get('stack_plots',True)

    n_plots = len(x_a)
    if bool_a is None: bool_a=n_plots*[True]
    if sty_a is None: sty_a=n_plots*['-']
    if alpha_a is None: alpha_a=n_plots*[1.0]
    if any(len(_a) != n_plots for _a in [y_a,bool_a]): raise Exception('ERROR: Inconsistent no. of plots in lists!')
    
    if fill and any(_a is None for _a in [std_a]):
        raise Exception('ERROR: Need std_a argument for "fill" option!')
    
    ############################# Calculate hspace per plot and ylim #################
    _slc = [slice(*sorted(np.argmin(np.abs(_x_a-_x)) for _x in xlim)) for _x_a in x_a]
    if _exp is not None: _slce = slice(*sorted(np.argmin(np.abs(xe-_x)) for _x in xlim))
    
    _shift=max([np.amax(_y[_s])-np.amin(_y[_s]) for _y,_s in zip(y_a,_slc)])
    if _exp is not None: _shift=max(np.amax(e[_slce])-np.amin(e[_slce]),_shift) 
    _shift *= (1+sep/100)
    print(_shift)

    if stack:
        _y_a = [_y-_shift*_i for _i,_y in enumerate(y_a)]
        if _exp is not None: _e = e-n_plots*_shift
    else:
        _y_a = y_a
        if _exp is not None: _e = e

    if ylim is None:
        ylim=(min([np.amin(_y[_s]) for _y,_s in zip(_y_a,_slc)]),max([np.amax(_y[_s]) for _y,_s in zip(_y_a,_slc)]))
        if _exp is not None: ylim = (min(ylim[0],np.amin(_e[_slce])),max(ylim[1],np.amax(_e[_slce])))
        ylim = (ylim[0]-0.25*_shift,ylim[1]+0.25*_shift)

    ############################# plot reference (experiment) #################
    if _exp is not None: ax.plot(xe,_e,'-',lw=3,color='black',label='exp.')
    
    
    ############################# plot data #############################    
    if fill:
        for _b,_x,_y,_st,_c,_al,_s in zip(bool_a,x_a,_y_a,sty_a,color_a,alpha_a,std_a):
            if _b: 
                ax.fill_between(_x, _y+_s, _y-_s, color=_c, alpha=.25)
                ax.plot(_x,_y,_st,lw=3,color=_c,alpha=_al)
    else:
        for _b,_x,_y,_st,_c,_al in zip(bool_a,x_a,_y_a,sty_a,color_a,alpha_a):
            if _b: ax.plot(_x,_y,_st,lw=3,color=_c,alpha=_al)
    

    ############################# layout ######################################
    make_nice_ax(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    ############################# export label object (beta) ###############
    ### LB is simply returned can be retrieved (or not)
    ### ToDO: Connect the tuple entries (so that e.g., X has only to be set once, but allow exceptional X values)
    class pub_label():
        def __init__(self,ax,**kwargs):
            self.ax = ax
            self.X = kwargs.get('X',1400)
            self.Y = kwargs.get('Y',0.0)
            self.color = kwargs.get('color','black')
            self.size = kwargs.get('size',26)
            self.stancil = kwargs.get('stancil',r'\textbf{%s}')
            self.sep = kwargs.get('sep',0.2*_shift)  ##ATTENTION: using varibale from outer scope
            self.alpha = kwargs.get('alpha',1.0)
        def print(self,string,**kwargs):
            X = kwargs.get('X',self.X)
            Y = kwargs.get('Y',self.Y)
            alpha=kwargs.get('alpha',self.alpha)
            self.ax.text(X,Y+self.sep,self.stancil%string,color=self.color,size=self.size,alpha=alpha)
            

    if stack: LB = [pub_label(ax,color=_c,X=np.mean(xlim),Y=-_shift*_i,alpha=_al) for _i,(_c,_al) in enumerate(zip(color_a,alpha_a))] \
                 + [pub_label(ax,color='black',X=np.mean(xlim),Y=-n_plots*_shift,stancil=r'\emph{%s}')]*(_exp is not None)
    else: LB = [pub_label(ax,color=_c,X=np.mean(xlim),alpha=_al) for _c,_al in zip(color_a,alpha_a)] \
             + [pub_label(ax,color='black',X=np.mean(xlim),stancil=r'\emph{%s}')]*(_exp is not None)

    return LB

