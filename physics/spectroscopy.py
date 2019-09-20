#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------
#!/usr/bin/env python3

import copy
import numpy as np

from ..physics import classical_electrodynamics as edyn
from ..physics import constants

#constants
h_si = 6.62606957E-34 # Planck's constant [Js]
hbar_si = h_si/(2*np.pi) # reduced planck constant [Js]
E_au = 4.35974417E-18 # Hartree energy [J]
time_au = hbar_si/E_au # time in A.U. [s]
fs2au = 1/time_au*1E-15
Angstrom2Bohr = 1.8897261247828971
eijk = np.zeros((3, 3, 3))
eijk[0,1,2] = eijk[1,2,0] = eijk[2,0,1] = +1
eijk[0,2,1] = eijk[2,1,0] = eijk[1,0,2] = -1

#### MOVE TO UTILS ###################################################################################################
   # change tp name standard: fermi_cutoff_function
def FermiCutoffFunction(distance_au, R_cutoff_au, D_au=0.25*constants.l_aa2au):
    return 1/(1+np.exp((distance_au - R_cutoff_au)/D_au))


#old python from Arne
def Filter(n_frames, filter_length=None, filter_type='welch'):
    if filter_length == None:
        filter_length = n_frames
    if filter_type == 'hanning':
        return np.hanning(2*filter_length)[n_frames:]
    elif filter_type == 'welch':
        return (np.arange(filter_length)[::-1]/(filter_length+1))**2
    elif filter_type == 'triangular':
        return np.ones(filter_length) - np.arange(filter_length)/n_frames
    else:
        raise Exception('Filter %s not supported!'%filter_type)

def CurrentCurrentPrefactor(temperature_K):
    prefactor_cgs = DipoleDipolePrefactor(temperature_K)*(Bohr_cgs/time_au)**2
    cm2_kmcm      = 1E-5
    return prefactor_cgs*cm2_kmcm*time_au

def CurrentMagneticPrefactor(nu_cgs, temperature_K):
    prefactor_cgs = DipoleDipolePrefactor(temperature_K)*(Bohr_cgs**3/time_au**2/speedoflight_cgs)
    cm2_kmcm      = 1E-5
    omega_cgs     = nu_cgs*speedoflight_cgs*2*np.pi
    return 4*omega_cgs*prefactor_cgs*cm2_kmcm*time_au

def DipoleDipolePrefactor(temperature_K):
    beta_cgs      = 1./(temperature_K*k_Boltzmann_cgs)
    prefactor_cgs = (2*np.pi*N_avogadro*beta_cgs*finestructure_constant*hbar_cgs)/3
    return prefactor_cgs

######################################################################################################################

def calculate_spectral_density(val1,*args,**kwargs):
    '''automatically does auto- or cross correlation, ignores 3rd *arg and beyond'''
    #n_frames,n_atoms,three = val1.shape
    n_frames,three = val1.shape
    ts = kwargs.get('ts',4) #time step in fs
    dt = ts*1E-15*1E9 #4.0 fs --> GHz
    flt_pow = kwargs.get('flt_pow',11)
    _fac = kwargs.get('factor',1)
    filter = Filter(n_frames, filter_length=n_frames, filter_type='welch')**flt_pow

    auto=True
    try: 
        val2 = args[0]
        auto = False
    except IndexError: 
        pass

    if auto:
        val_ac = np.array([np.correlate(v1,v1,mode='full')[n_frames-1:] for v1 in val1.T]).T 
    else:
        val_ac  = np.array([np.correlate(v1,v2,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T 
        val_ac += np.array([np.correlate(v2,v1,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T 
        val_ac /= 2 #mu(o).m(t) - m(0).mu(t) for ergodic systems the two contributions are equal

    val_ac /= (n_frames*np.ones(n_frames)-np.arange(n_frames))[:,None] # finite size
    val_ac = val_ac.sum(axis=1)
    val_ac *= filter
    final_cc = np.hstack((val_ac,val_ac[::-1]))
    n = final_cc.shape[0]

    w1 = np.fft.rfft(final_cc,n=n-1).real*ts*fs2au*_fac
    x_spec = np.fft.rfftfreq(n-1,d=dt)

    return x_spec,w1

def get_power_spectrum(val,**kwargs):
    n_frames,n_atoms,three = val.shape
    wgh = kwargs.get('weights',np.ones(n_atoms))
    x_spec,w1 =  zip(*[calculate_spectral_density(_v,**kwargs) for _v in val.swapaxes(0,1)])

    return x_spec[0],(np.array(w1)*wgh[:,None]).sum(axis=0)

def get_ira_and_vcd(c,m,pos_au,**kwargs):  
    origins_au = kwargs.get('origins_au',np.array([[0.0,0.0,0.0]]))
    cell_au = kwargs.get('cell_au')
    #cell_au = kwargs.get('cell_aa')*Angstrom2Bohr
    cutoff_aa = kwargs.get('cutoff_aa',0)
    cut_type = kwargs.get('cut_type','soft')
    subparticle = kwargs.get('subparticle') #beta: correlate moment of one part with total moments, expects one integer
    subparticles = kwargs.get('subparticles') #beta: correlate moment of one part with another part, expects tuple of integers
    subnotparticles = kwargs.get('subnotparticles') #beta: correlate moment of one part with another part, notparticles: exclude these indices; expects list of integers
    cutoff_bg_aa = kwargs.get('background_correction_cutoff_aa') #beta: remove direct correlation between background 

    spectrum = list()
    for _i,_o in enumerate(origins_au):
        _c = copy.deepcopy(c)
        _m = copy.deepcopy(m)

        #R_I(t) - R_J(t) #used in manuscript
        _trans = pos_au-_o[:,None,:] 
        if cell_au is not None:
            _trans -= np.around(_trans/cell_au)*cell_au
        #print(_trans[0])

        #R_I(t) - R_J(0) #gauge invariant according to Rodolphe/Arne
        #_trans = pos_au-_o[0,None,None,:] 
        #_trans -= np.around(_trans/cell_au)*cell_au        
        #_trans += _o[:,None,:] - _o[0,None,None,:]
        _m += edyn.magnetic_dipole_shift_origin(_c,_trans)
        #m += 0.5*np.sum(eijk[None,None,:,:,:]*trans[:,:,:,None,None]*c[:,:,None,:,None], axis=(2,3)) 


        if cut_type=='soft': #for larger cutoffs
            _scal = FermiCutoffFunction(np.linalg.norm(_trans,axis=2), cutoff_aa*Angstrom2Bohr,D_au=0.125*constants.l_aa2au)
            _c *= _scal[:,:,None]
            _m *= _scal[:,:,None]

        if cut_type=='hard': #cutoff <2 aa
            _ind  = np.linalg.norm(_trans,axis=2)>cutoff_aa*Angstrom2Bohr
            _c[_ind ,:] = np.array([0.0,0.0,0.0])
            _m[_ind ,:] = np.array([0.0,0.0,0.0])

        if type(cutoff_bg_aa) is float:
            _c_bg = copy.deepcopy(_c)
            #_m_bg = copy.deepcopy(m) #only direct correlation, no transport term!
            _m_bg = copy.deepcopy(_m) #complete background
            if cut_type=='soft': _m_bg *= _scal[:,:,None] 
            if cut_type=='hard': _m[_ind ,:] = np.array([0.0,0.0,0.0])
            _ind_bg  = np.linalg.norm(_trans,axis=2)<=cutoff_bg_aa*Angstrom2Bohr #bg cut is always hard
            _c_bg[_ind_bg ,:] = np.array([0.0,0.0,0.0])
            _m_bg[_ind_bg ,:] = np.array([0.0,0.0,0.0])

        if all([subparticle is None,subparticles is None,subnotparticles is None]):
            _c = _c.sum(axis=1)
            _m = _m.sum(axis=1)
            x_spec,ira = calculate_spectral_density(_c,**kwargs)
            x_spec,vcd = calculate_spectral_density(_c,_m,**kwargs)
            if type(cutoff_bg_aa) is float:
                _c_bg = _c_bg.sum(axis=1)
                _m_bg = _m_bg.sum(axis=1)
                ira -= calculate_spectral_density(_c_bg,**kwargs)[1]
                vcd -= calculate_spectral_density(_c_bg,_m_bg,**kwargs)[1]

        elif type(subparticle) is int:
            _c1 = _c.sum(axis=1)
            _c2 = _c[:,subparticle]
            _m1 = _m.sum(axis=1)
            _m2 = _m[:,subparticle]
            x_spec,ira  = calculate_spectral_density(_c1,_c2,**kwargs)
            x_spec,vcd1 = calculate_spectral_density(_c1,_m2,**kwargs)
            x_spec,vcd2 = calculate_spectral_density(_c2,_m1,**kwargs)
            vcd = (vcd1+vcd2)/2
            if type(cutoff_bg_aa) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = _c_bg[:,subparticle]
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = _m_bg[:,subparticle]
                ira -= calculate_spectral_density(_c1_bg,_c2_bg,**kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(_c1_bg,_m2_bg,**kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(_c2_bg,_m1_bg,**kwargs)[1]

        elif type(subnotparticles) is list:
            _c1 = _c.sum(axis=1)
            _c2 = copy.deepcopy(_c)
            _c2[:,subnotparticles] = np.array([0.0,0.0,0.0])
            _c2 = _c2.sum(axis=1)
            _m1 = _m.sum(axis=1)
            _m2 = copy.deepcopy(_m)
            _m2[:,subnotparticles] = np.array([0.0,0.0,0.0])
            _m2 = _m2.sum(axis=1)
            x_spec,ira  = calculate_spectral_density(_c1,_c2,**kwargs)
            x_spec,vcd1 = calculate_spectral_density(_c1,_m2,**kwargs)
            x_spec,vcd2 = calculate_spectral_density(_c2,_m1,**kwargs)
            vcd = (vcd1+vcd2)/2
            if type(cutoff_bg_aa) is float:
                _c1_bg = _c_bg.sum(axis=1)
                _c2_bg = copy.deepcopy(_c_bg)
                _c2_bg[:,subnotparticles] = np.array([0.0,0.0,0.0])
                _c2_bg = _c2_bg.sum(axis=1)
                _m1_bg = _m_bg.sum(axis=1)
                _m2_bg = copy.deepcopy(_m_bg)
                _m2_bg[:,subnotparticles] = np.array([0.0,0.0,0.0])
                _m2_bg = _m2_bg.sum(axis=1)
                ira -= calculate_spectral_density(_c1_bg,_c2_bg,**kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(_c1_bg,_m2_bg,**kwargs)[1]
                vcd -= 0.5*calculate_spectral_density(_c2_bg,_m1_bg,**kwargs)[1]

        elif type(subparticles) is tuple:
            _c1 = _c[:,subparticles[0]]
            _c2 = _c[:,subparticles[1]]
            _m1 = _m[:,subparticles[0]]
            _m2 = _m[:,subparticles[1]]
            x_spec,ira  = calculate_spectral_density(_c1,_c2,**kwargs)
            x_spec,vcd1 = calculate_spectral_density(_c1,_m2,**kwargs)
            x_spec,vcd2 = calculate_spectral_density(_c2,_m1,**kwargs)
            vcd = (vcd1+vcd2)/2
            if type(cutoff_bg_aa) is float: raise Exception('ERROR: Background correction not implemented for subparticles option!')

        else: raise Exception('ERROR: subparticle(s) is (are) not an integer (a tuple of integers)!')

        _cc = CurrentCurrentPrefactor(300)
        _cm = CurrentMagneticPrefactor(x_spec,300)
        spectrum.append([x_spec,ira*_cc,vcd*_cm])
    spectrum = np.array(spectrum).sum(axis=0)/origins_au.shape[0]
    
    return spectrum

################## REPAIR and DEBUG #######################################################
def Get_IR_and_VCD_NoBackground(val1a,val2a,pos_au,origins_au,cell_au,cutoff_aa,IR=False):  
    spectrum_ac = list()
    n_frames,n_atoms,three = val1a.shape
    for i,o in enumerate(origins_au):
        val1 = copy.deepcopy(val1a)
        val2 = copy.deepcopy(val2a)
        
        #R_I(t) - R_J(t) #used in manuscript
        v_trans_au = pos_au-o[:,None,:] 
        v_trans_au -= np.around(v_trans_au/cell_au)*cell_au
     
    
        #softer
        scal = FermiCutoffFunction(np.linalg.norm(v_trans_au,axis=2), cutoff_aa*Angstrom2Bohr,D_au=0.5*constants.l_aa2au)
        #ind  = np.linalg.norm(v_trans_au,axis=2)>cutoff_aa*Angstrom2Bohr
        ind_bg = np.linalg.norm(v_trans_au,axis=2)<=2.5#cutoff two for background removal and direct correlation
        #ind_bg = np.linalg.norm(v_trans_au,axis=2)>0
        
        val2 += 0.5*np.sum(eijk[None,None,:,:,:]*v_trans_au[:,:,:,None,None]*val1[:,:,None,:,None], axis=(2,3)) 
        #val2 = 0.5*np.sum(eijk[None,None,:,:,:]*v_trans_au[:,:,:,None,None]*val1[:,:,None,:,None], axis=(2,3)) 
        
        #val1[ind ,:] = np.array([0.0,0.0,0.0])
        #val2[ind ,:] = np.array([0.0,0.0,0.0])
        val1 *= scal[:,:,None]
        val2 *= scal[:,:,None]
        
        val1_bg = copy.deepcopy(val1)
        val2_bg = copy.deepcopy(val2a)
        val2_bg *= scal[:,:,None]
               
        val1_bg[ind_bg ,:] = np.array([0.0,0.0,0.0])
        val2_bg[ind_bg ,:] = np.array([0.0,0.0,0.0])
        
        val1    = val1.sum(axis=1) 
        val1_bg = val1_bg.sum(axis=1) 
        
        #IR=True
        if IR: #quick check only
            val2    = copy.deepcopy(val1)
            val2_bg = copy.deepcopy(val1_bg)
        else:
            val2    = val2.sum(axis=1) 
            val2_bg = val2_bg.sum(axis=1) 
                   
        val_ac = np.array([np.correlate(v1,v2,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T #For autocorrelations (cross needs zip ...)
        val_ac += np.array([np.correlate(v2,v1,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T #For autocorrelations (cross needs zip ...)
        val_ac /= 2 #mu(o).m(t) - m(0).mu(t) for ergodic systems the two contributions are equal
        
        #Background correction
        if True:
            val_ac -= 0.5*np.array([np.correlate(v1,v2,mode='full')[n_frames-1:] for v1, v2 in zip(val1_bg.T, val2_bg.T)]).T #For autocorrelations (cross needs zip ...)
            val_ac -= 0.5*np.array([np.correlate(v2,v1,mode='full')[n_frames-1:] for v1, v2 in zip(val1_bg.T, val2_bg.T)]).T #For autocorrelations (cross needs zip ...)
        
        val_ac /= (n_frames*np.ones(n_frames)-np.arange(n_frames))[:,None] # finite size
        val_ac = val_ac.sum(axis=1)
        filter = Filter(n_frames, filter_length=n_frames, filter_type='welch')
        val_ac *= filter**11

        final_cc = np.hstack((val_ac,val_ac[::-1]))
        n = final_cc.shape[0]
        w1 = np.fft.rfft(final_cc,n=n-1).real*ts*fs2au 
        x_spec = np.fft.rfftfreq(n-1,d=dt)
        spectrum_ac.append(w1)
    
    spectrum_ac = np.array(spectrum_ac).sum(axis=0)/origins_au.shape[0]
    if IR:
        cc = CurrentCurrentPrefactor(300)
    else:
        cc = CurrentMagneticPrefactor(x_spec,300)
    return [x_spec,spectrum_ac * cc ]

def Get_IR_and_VCD_NoCentre(val1a,val2a,pos_au,origins_au,cell_au,cutoff_aa,IR=False):  
    spectrum_ac = list()
    n_frames,n_atoms,three = val1a.shape
    for i,o in enumerate(origins_au):
        val1 = copy.deepcopy(val1a)
        val2 = copy.deepcopy(val2a)
        
        #R_I(t) - R_J(t) #used in manuscript
        v_trans_au = pos_au-o[:,None,:] 
        v_trans_au -= np.around(v_trans_au/cell_au)*cell_au
     
        scal = FermiCutoffFunction(np.linalg.norm(v_trans_au,axis=2), cutoff_aa*Angstrom2Bohr)

        val2 += 0.5*np.sum(eijk[None,None,:,:,:]*v_trans_au[:,:,:,None,None]*val1[:,:,None,:,None], axis=(2,3)) 
        
        val1 *= scal[:,:,None]
        val2 *= scal[:,:,None]
        
        #if i==0: 
        #    val1[:,1] *= 0.0
        #    val2[:,1] *= 0.0
        #elif i==1:
        #    val1[:,0] *= 0.0
        #    val2[:,0] *= 0.0
        val1[:,:2] *= 0.0
        val2[:,:2] *= 0.0        
        
        val1 = val1.sum(axis=1)
        
        #IR=True
        if IR: #quick check only
            val2 = copy.deepcopy(val1)
        else:
            val2 = val2.sum(axis=1)
                   
        val_ac = np.array([np.correlate(v1,v2,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T #For autocorrelations (cross needs zip ...)
        val_ac += np.array([np.correlate(v2,v1,mode='full')[n_frames-1:] for v1, v2 in zip(val1.T, val2.T)]).T #For autocorrelations (cross needs zip ...)
        val_ac /= 2 #mu(o).m(t) - m(0).mu(t) for ergodic systems the two contributions are equal
        
        val_ac /= (n_frames*np.ones(n_frames)-np.arange(n_frames))[:,None] # finite size
        val_ac = val_ac.sum(axis=1)
        filter = Filter(n_frames, filter_length=n_frames, filter_type='welch')
        val_ac *= filter**11

        final_cc = np.hstack((val_ac,val_ac[::-1]))
        n = final_cc.shape[0]
        w1 = np.fft.rfft(final_cc,n=n-1).real*ts*fs2au 
        x_spec = np.fft.rfftfreq(n-1,d=dt)
        spectrum_ac.append(w1)
    
    spectrum_ac = np.array(spectrum_ac).sum(axis=0)/origins_au.shape[0]
    if IR:
        cc = CurrentCurrentPrefactor(300)
    else:
        cc = CurrentMagneticPrefactor(x_spec,300)
    return [x_spec,spectrum_ac * cc ]

