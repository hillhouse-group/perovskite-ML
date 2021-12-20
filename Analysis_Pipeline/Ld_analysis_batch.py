# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:24:58 2019

@author: Ryan Stoddard
"""

#standard imports
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append('../')


import SQ_calcs
from stats import Constants


#change default plot settings
default_figsize = mpl.rcParamsDefault['figure.figsize']
mpl.rcParams['figure.figsize'] = [1.5*val for val in default_figsize]
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 24}

mpl.rc('font', **font)
mpl.rc('axes', linewidth=3)

#Constants
pi = math.pi
const = Constants()

# Experimental Parameters - Change these as necessary
Eg = const.sample_bandgap #eV, this is reference Eg for setting laser settings
photodiode_responsivity = 0.3 # A/W, approximate responsivity of UV-100DQ photodiode at 550 nm

# Relationship between Lumencor power setting (on 255 scale) and number of suns in the beam when using the 100X objective
# keys: Lumencor settings; values = Nsuns
sun_dict = {15:1,27:2,52:4,106:8,14:16,24:32,45:64,90:128,255:8,70:8,168:8,12:1,78:8,81:8,205:8,253:8,164:8}


def electronic_data_calculations(file_path, write_path, temp, film_thickness, 
                                 channel_length, channel_width, meas_cycle_time, 
                                 transmissivity_flag, sun_params, grad=0, loc=0):
    '''
    This function performs all of the data analysis steps from pymeasure data aquition (DC photoconductivity, AC PC, and Transmissivity)
    INPUTS:
        file_path: path to folder containing data
        write_path: path to folder to save plots
        temp: temperature [K]
        film_thickness: film thickness [m]
        channel_length: distance between contacts [m]
        channel_width:  width of contact pad (or beam) [m]
        meas_cycle_time: how frequently data cycles/primary videos were collected [min]
        transmissivity_flag: True or False (was T data collected)?

    RETURNS:
        df of all analyzed data (AC Ld, DC Ld, transmissivity, etc.)

    also generatures Ld plots in separate out folder
    '''


    T = temp # K, temperature
    d = film_thickness # m, film thickness
    l = channel_length # m, distance between contacts - i.e., channel length
    w = channel_width # m, width of contact pad (or beam) - i.e., channel width
    DAQ_period = meas_cycle_time # min, how frequently data cycles/primary videos were collected



    WriteDir = os.path.join(write_path, 'Ld_plots')


    name = [os.path.join(file_path, 'electronic_measurement_data.csv')]

    # Useful functions
    def Ld_func(sigma, T, G):
     #Ld in nm
     Ld = 1e9*math.sqrt(sigma*const.k*T/(2*const.q**2*G))
     return Ld
     
     # convert Lumencor 255 scale readings to Nsuns
    def L255_to_Nsuns(setting,params):
        nsuns = params[0]*setting**2 + params[1]*setting + params[0]
        return nsuns


    # Create array of incident flux for each measurement
    Incident_flux = SQ_calcs.one_sun_photon_flux(Eg)
    G_1sun = Incident_flux/d # #/m^3/sec, carrier generation rate
    #print(Incident_flux)

    data = pd.read_csv(name[0],header=None)
    # Remove ending zeros from data df (this happens when experiment was terminated in the middle of a cycle)
    kk = 0
    while data.iloc[-kk,0] == 0:
        kk+=1
    if kk > 1:
        data = data.iloc[:-kk+1,:]
    
    
    # Assign names to the columns of the dataframe, If no transmissivitity, just add col of zeros
    if transmissivity_flag:
        data.columns = ['primary video index','gradient index','position index','time (sec)','net current from Keithley (A)','light current from SRS (A)','source voltage (V)','Lumencor power setting','strobe frequency (Hz)','photodiode current (A)']
    else:
        data.columns = ['primary video index','gradient index','position index','time (sec)','net current from Keithley (A)','light current from SRS (A)','source voltage (V)','Lumencor power setting','strobe frequency (Hz)']
        data['photodiode current (A)'] = 0

    
    # Only use data for particular grad and loc
    data = data[data['gradient index'] == grad]
    data = data[data['position index'] == loc]
    

    # Determine how many primary videos were taken
    cycles=data['primary video index'].values

    Ncycles = cycles[-1] + 1


    # Determine how many measurements over the course of one photoconductivity measurement
    collection_length = len(cycles)/Ncycles


    # Get number of fields in the raw dataset
    Nfields = len(data.columns)

    # Fix issue than can arize when experiment interrupted at an awkward place
    #data = data.iloc[:int(Ncycles)*int(collection_length),:]
    
    # It's aggravating
    # Reshape: pages, rows, columns
    # But what can you do
    data_3D = data.values.reshape(int(Ncycles),int(collection_length),Nfields)

    # Declare variables of interest

    DC_Photoconductance = np.zeros(int(Ncycles)) # S, DC photoconductance
    AC_Photoconductance_low = np.zeros(int(Ncycles)) # S, low-frequency AC photoconductance
    AC_Photoconductance_high = np.zeros(int(Ncycles)) # S, high-frequency AC photoconductance
    Transmitted_Power = np.zeros(int(Ncycles)) # W, optical power transmitted through sample

    DC_sigma = np.zeros(int(Ncycles)) # S/m, DC light conductivity
    AC_sigma_lo = np.zeros(int(Ncycles)) # S/m, low-frequency AC light conductivity
    AC_sigma_hi = np.zeros(int(Ncycles)) # S/m, high-frequency AC light conductivity


    Ld_DC = np.zeros(int(Ncycles)) # nm, diffusion length
    Ld_AC_lo = np.zeros(int(Ncycles)) # nm, diffusion length
    Ld_AC_hi = np.zeros(int(Ncycles)) # nm, diffusion length

    Nsuns = np.zeros(int(Ncycles)) # suns, intensity

    # This loop operates over all the primary videos; index jj corresponds to an individual PLVA/Ld data collection cycle
    for jj in range(int(Ncycles)):

        V_appl = data_3D[jj,0,6] # get applied bias [V]

        # Determine how many data points were sampled per frequency condition (should be the same for all of them)
        pts_sampled_per_freq_mode = int(collection_length/4)

        # Initialize dark, DC light, low frequency, and high frequency AC light currents
        I_dark = np.zeros(pts_sampled_per_freq_mode)
        I_light_DC = np.zeros(pts_sampled_per_freq_mode)
        I_light_lo_f = np.zeros(pts_sampled_per_freq_mode)
        I_light_lo_f_Keithley = np.zeros(pts_sampled_per_freq_mode)
        I_light_hi_f = np.zeros(pts_sampled_per_freq_mode)

        # Initialize dark and light transmitted power variables
        dark_opt_power = np.zeros(pts_sampled_per_freq_mode)
        light_opt_power = np.zeros(pts_sampled_per_freq_mode)

        low_freq = data_3D[jj,2*pts_sampled_per_freq_mode,8]
        #print('Low strobe frequency [Hz]:')
        #print(low_freq)
        high_freq = data_3D[jj,3*pts_sampled_per_freq_mode,8]
        #print('High strobe frequency [Hz]:')
        #print(high_freq)

        # This loop operates over time within an individual DAQ cycle: index kk corresponds to time
        # Assign different portions of the Keithley and SRS time series to different DAQ modes
        for kk in range(int(collection_length)):
            strobe_freq = data_3D[jj,kk,8] # get strobe frequency

            # sort current data by data acqusition mode:
            if np.isnan(strobe_freq): # if strobe frequency is Not a Number, assign to dark current
                I_dark[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,4]
                dark_opt_power[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,9]/photodiode_responsivity

            elif strobe_freq == 0: # if strobe frequency is zero, assign to DC light current
                I_light_DC[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,4]
                light_opt_power[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,9]/photodiode_responsivity

            elif strobe_freq == low_freq: # if strobe frequency is low, assign to low-frequency AC light current
                I_light_lo_f_Keithley[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,4]
                I_light_lo_f[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,5]

            elif strobe_freq == high_freq: # if strobe frequency is high, assign to high-frequency AC light current
                I_light_hi_f[kk % pts_sampled_per_freq_mode] = data_3D[jj,kk,5]


        # Take the dark current as the average of the last ten points in the time series
        darkCurrent = np.mean(I_dark[len(I_dark)-10:len(I_dark)])
        #print('The dark current [A] is:')
        #print(darkCurrent)
        darkPower = np.mean(dark_opt_power[len(dark_opt_power)-10:len(dark_opt_power)])

        # Take the DC light current as the average of the last ten points in the time series
        lightCurrent_DC = np.mean(I_light_DC[len(I_light_DC)-10:len(I_light_DC)])
        #print('The DC light current [A] is:')
        #print(lightCurrent_DC)
        lightPower = np.mean(light_opt_power[len(light_opt_power)-10:len(light_opt_power)])

        # Subtract the dark from light current to get the net photocurrent
        DC_photocurrent = lightCurrent_DC - darkCurrent
        #print('The DC photcurrent [A] is:')
        #print(DC_photocurrent)

        # Subtract background light to get net transmitted optical power through sample
        Transmitted_Power[jj] = lightPower-darkPower

        # Take the AC light current as the average of the last ten points in the time series
        AC_photocurrent_lo_freq = np.mean(I_light_lo_f[len(I_light_lo_f)-10:len(I_light_lo_f)])
        #print('The low-frequency light current [A] is:')
        #print(AC_photocurrent_lo_freq)

        # Take the AC light current as the average of the last ten points in the time series
        AC_photocurrent_hi_freq = np.mean(I_light_hi_f[len(I_light_hi_f)-10:len(I_light_hi_f)])
        #print('The high-frequency light current [A] is:')
        #print(AC_photocurrent_hi_freq)


        # Convert photocurrent measurements to photoconductances
        DC_Photoconductance[jj] = DC_photocurrent/V_appl
        AC_Photoconductance_low[jj] = AC_photocurrent_lo_freq/V_appl
        AC_Photoconductance_high[jj] = AC_photocurrent_hi_freq/V_appl

        # Convert photoconductance measurements to diffusion lengths
        DC_sigma[jj] = np.absolute(DC_Photoconductance[jj]*(l/(d*w))) # S/m, DC light conductivity
        AC_sigma_lo[jj] = np.absolute(AC_Photoconductance_low[jj]*(l/(d*w))) # S/m, low-frequency AC light conductivity
        AC_sigma_hi[jj] = np.absolute(AC_Photoconductance_high[jj]*(l/(d*w))) # S/m, high-frequency AC light conductivity

        #Nsuns[jj] = sun_dict[np.max(data_3D[jj,:,7])]
        Nsuns[jj] = L255_to_Nsuns(np.max(data_3D[jj,:,7]),sun_params)
        #Nsuns[jj] = 1

        Ld_DC[jj] = Ld_func(DC_sigma[jj], T, G_1sun*Nsuns[jj])
        Ld_AC_lo[jj] = Ld_func(AC_sigma_lo[jj], T, G_1sun*Nsuns[jj])
        Ld_AC_hi[jj] = Ld_func(AC_sigma_hi[jj], T, G_1sun*Nsuns[jj])



    sun_setting = Nsuns[0]

    time_range = np.array(range(int(Ncycles)))*DAQ_period

    # Make and save plots

    if not os.path.exists(WriteDir):
        os.makedirs(WriteDir)

    plt.figure()
    ax=plt.plot(time_range, Ld_DC, '.', markersize=12, label='DC')
    plt.ylabel('$L_D\ [nm]$')
    plt.xlabel('$Elapsed\ Time\ [min]$')
    ax=plt.plot(time_range, Ld_AC_lo, '.', markersize=12, label='AC - low f')
    ax=plt.plot(time_range, Ld_AC_hi, '.', markersize=12, label='AC - high f')
    plt.legend()
    plt.savefig(os.path.join(WriteDir, 'Ld.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    ax=plt.plot(time_range, Ld_DC/Ld_DC[0], '.', markersize=12, label='DC')
    ax=plt.plot(time_range, Ld_AC_lo/Ld_AC_lo[0], '.', markersize=12, label='AC - low f')
    ax=plt.plot(time_range, Ld_AC_hi/Ld_AC_hi[0], '.', markersize=12, label='AC - high f')
    plt.ylabel('$Normalized\ L_D$')
    plt.xlabel('$Elapsed\ Time\ [min]$')
    plt.legend()
    plt.savefig(os.path.join(WriteDir, 'Ld_norm.png'), dpi=300, bbox_inches='tight')
    plt.close()
  
    if transmissivity_flag:
        plt.figure()
        ax=plt.plot(time_range, Transmitted_Power*1e6, '.', markersize=12, label='DC')
        plt.ylabel('$Transmitted\ Optical\ Power\ [\mu W]$')
        plt.xlabel('$Elapsed\ Time\ [min]$')
        plt.savefig(os.path.join(WriteDir, 'Transmissivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure()
        ax=plt.plot(time_range, (Transmitted_Power/Transmitted_Power[0]), '.', markersize=12, label='DC')
        plt.ylabel('$Normalized\ Transmitted\ Optical\ Power$')
        plt.xlabel('$Elapsed\ Time\ [min]$')
        plt.savefig(os.path.join(WriteDir, 'Transmissivity_norm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        df = pd.DataFrame({'Index':range(int(Ncycles)), 'DC LD [nm]':Ld_DC, 'Low Freq LD [nm]':Ld_AC_lo, 'High Freq LD [nm]':Ld_AC_hi, 'Transmitted Power [W]':Transmitted_Power,
                           'DC LD [norm]':Ld_DC/Ld_DC[0],  'Low Freq LD [norm]':Ld_AC_lo/Ld_AC_lo[0], 'High Freq LD [norm]':Ld_AC_hi/Ld_AC_hi[0], 'Transmitted Power [norm]':Transmitted_Power/Transmitted_Power[0]})
    else:
        df = pd.DataFrame({'Index':range(int(Ncycles)), 'DC LD [nm]':Ld_DC, 'Low Freq LD [nm]':Ld_AC_lo, 'High Freq LD [nm]':Ld_AC_hi,
                           'DC LD [norm]':Ld_DC/Ld_DC[0],  'Low Freq LD [norm]':Ld_AC_lo/Ld_AC_lo[0], 'High Freq LD [norm]':Ld_AC_hi/Ld_AC_hi[0]})
        
    return df, sun_setting
#df.to_csv(read_write_path + 'Ld_timeseries.csv')
