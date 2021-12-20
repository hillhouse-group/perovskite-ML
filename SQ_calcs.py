import numpy as np
import pandas as pd
import math
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import signal


import matplotlib.pyplot as plt

#Constants
pi = math.pi
heV = 4.14e-15 #eV*s
c = 2.99792e8 #m/s
kbeV = 8.6173e-5 #eV/K
keV = 8.6173e-5 #eV/K
h = 6.626e-34
kb = 1.38065e-23
q = 1.60218e-19
k = 1.3806488e-23
JperEV = 1.60218e-19
T = 293
Fs = 0.0000680024

#Load data

try:
    am15_nm = np.loadtxt('../../hp-db/confocal-pl//vocmax_data/am15_nmdata.txt', delimiter = ',')
    am15_energy_pernm = np.loadtxt('../../hp-db/confocal-pl/vocmax_data/am15_spec_data.txt', delimiter = ',')
except OSError:
    try:
        am15_nm = np.loadtxt('hp-db/confocal-pl/vocmax_data/am15_nmdata.txt', delimiter = ',')
        am15_energy_pernm = np.loadtxt('hp-db/confocal-pl/vocmax_data/am15_spec_data.txt', delimiter = ',')
    except OSError:
        try:
            am15_nm = np.loadtxt('../hp-db/confocal-pl/vocmax_data/am15_nmdata.txt', delimiter = ',')
            am15_energy_pernm = np.loadtxt('../hp-db/confocal-pl/vocmax_data/am15_spec_data.txt', delimiter = ',')
        except:
            am15_nm = np.loadtxt('C:/Users/Wiley.LAPTOP-KQ1TRN8R/Desktop/Just In Case/Calculations/hp-db/hp-db/confocal-pl/vocmax_data/am15_nmdata.txt', delimiter = ',')
            am15_energy_pernm = np.loadtxt('C:/Users/Wiley.LAPTOP-KQ1TRN8R/Desktop/Just In Case/Calculations/hp-db/hp-db/confocal-pl/vocmax_data/am15_spec_data.txt', delimiter = ',')

import_folder = 'hp-db/confocal-pl/vocmax_data/'
try:
    Egs = np.loadtxt(import_folder + 'Egs.txt',delimiter=',')
    VocSQs300 = np.loadtxt(import_folder + 'VocMaxs.txt',delimiter=',') # 300 K
    Jphs = np.loadtxt(import_folder + 'Jphs.txt',delimiter=',') #300 K
except OSError:
    try:
        Egs = np.loadtxt('../' + import_folder + 'Egs.txt',delimiter=',')
        VocSQs300 = np.loadtxt('../' + import_folder + 'VocMaxs.txt',delimiter=',') # 300 K
        Jphs = np.loadtxt('../' + import_folder + 'Jphs.txt',delimiter=',') #300 K
    except OSError:
        try:
            Egs = np.loadtxt('../../' + import_folder + 'Egs.txt',delimiter=',')
            VocSQs300 = np.loadtxt('../../' + import_folder + 'VocMaxs.txt',delimiter=',') # 300 K
            Jphs = np.loadtxt('../../' + import_folder + 'Jphs.txt',delimiter=',') #300 K
        except:
            Egs = np.loadtxt('C:/Users/Wiley.LAPTOP-KQ1TRN8R/Desktop/Just In Case/Calculations/hp-db/hp-db/confocal-pl/vocmax_data/Egs.txt',delimiter=',')
            VocSQs300 = np.loadtxt('C:/Users/Wiley.LAPTOP-KQ1TRN8R/Desktop/Just In Case/Calculations/hp-db/hp-db/confocal-pl/vocmax_data/VocMaxs.txt',delimiter=',') # 300 K
            Jphs = np.loadtxt('C:/Users/Wiley.LAPTOP-KQ1TRN8R/Desktop/Just In Case/Calculations/hp-db/hp-db/confocal-pl/vocmax_data/Jphs.txt',delimiter=',') #300 K
    
am15_ev = heV * c/ 1e-9 / am15_nm
    
#calculate photon flux at one sun for different bandgap
def one_sun_photon_flux(bandgap):
    '''
    This function integrates above bandgap photon flux for AM1.5GT Spectrum
    INPUTS:
    bandgap = bandgap [eV]
    
    OUTPUTS:
    above_bandgap_photon_flux = integrated above bandgap photon flux [photons/m^2-s]
    '''
    # This function inputs bandgap [eV] and returns above bandgap photon flux [photons/m^2-s] from AM1.5GT spectrum
    am15_idx = np.argmin(np.abs(am15_ev-bandgap))
    am15_photon_perev = AM15GTPhotonFluxPerEV(am15_ev)
    above_bandgap_photon_flux = -trapz(am15_photon_perev[:am15_idx],am15_ev[:am15_idx]) #photons/m^2-sec
    return above_bandgap_photon_flux
    
def AM15GTPhotonFluxPerEV(eV):
    '''
    This function does bandwith conversion for AM1.5GT Spectrum
    INPUTS:
    eV = energy [eV]
    
    OUTPUTS:
    photon_flux = photon flux per eV [photons/m^2-s-eV]
    '''
    AM15GT_fun = interp1d(am15_nm,am15_energy_pernm,'cubic',fill_value=0,bounds_error=False)
    photon_flux = (1/(eV*JperEV))*AM15GT_fun(h*c*1e9/(eV*JperEV))*(h*c/(eV*JperEV)**2)*1e9*JperEV
    return photon_flux

def JphSQ(Eg, Ta):
    '''
    This functions calculates the Shockley-Quiesser limit short curcuit current
    INPUTS:
    Eg = bandgap [eV]
    Ta = Temperature [K]
    
    OUTPUTS:
    J = short-circuit current [A/m^2]
    '''
    dm = 0 # delta mu
    beEV = (2*pi/(heV**3*c**2)*((am15_ev)**2/(np.exp((am15_ev-dm)/(keV*Ta))-1)))
    Japh = AM15GTPhotonFluxPerEV(am15_ev) - (Fs/pi)*beEV
    am15_idx = np.argmin(np.abs(am15_ev-Eg))
    J = -q*trapz(Japh[:am15_idx], am15_ev[:am15_idx])
    return J

def VocSQ(Eg, Ta):
    '''
    This functions calculates the Shockley-Quiesser limit open-circuit voltage
    INPUTS:
    Eg = bandgap [eV]
    Ta = Temperature [K]
    
    OUTPUTS:
    V = open-circuit voltage [V]
    '''
    
    dm = 0 # delta mu
    beEV = (2*pi/(h**3*c**2)*((am15_ev*JperEV)**2/(np.exp((am15_ev-dm)/(keV*Ta))-1))*JperEV)
    beEV_dm = np.zeros(am15_ev.shape)
    dm = Eg/2
    for ii in range(am15_ev.shape[0]):
        if am15_ev[ii] > dm:
            beEV_dm[ii] = (2*pi/(h**3*c**2)*((am15_ev[ii]*JperEV)**2/(np.exp((am15_ev[ii]-dm)/(keV*Ta))-1))*JperEV)  
    Jeph = (beEV_dm-beEV)    
    am15_idx = np.argmin(np.abs(am15_ev-Eg))    
    JdarkSQ = -q*trapz(Jeph[:am15_idx],am15_ev[:am15_idx])
    JoSQ = JdarkSQ/(np.exp(q*Eg/2/(k*Ta))-1)    
    V = k*Ta/q * np.log(JphSQ(Eg, Ta)/JoSQ + 1)    
    return V
    
def VocMax(Eg, NSuns, Ta):
    '''
    This functions Max Voc from S-Q limit, generalized for any Number of Suns
    
    This function assumes T = 300 K
    
    This fuction optimizes for speed by loading pre-computed VocSQ data,
    interpolating, and only computing VocSQ integrals if outside of data range
    
    INPUTS:
    Eg = bandgap [eV]
    NSuns = NSuns, 1 = integrated above Eg photon flux AM1.5GT
    
    OUTPUTS:
    V = max open-circuit voltage [V]
    '''
    Ta = 300
    
    VocSQs300_fn = interp1d(Egs, VocSQs300, kind='cubic')
    Jphs_fn = interp1d(Egs, Jphs, kind='cubic')
    try:
        VocSQ300_1Sun = VocSQs300_fn(Eg)
    except ValueError:
        VocSQ300_1Sun = VocSQ(Eg, Ta)
    
    V = VocSQ300_1Sun + keV*Ta*np.log(NSuns)
    return V
    
    
    
    
    