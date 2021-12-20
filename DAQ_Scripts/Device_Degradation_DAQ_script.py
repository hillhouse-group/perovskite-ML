# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:20:06 2019

@author: Ryan Stoddard
"""

'''
Procedure:
    1. Turn on Lumencor, Agilent, keithley,
    2. Check connections
        b. Keithley both leads to device
        c. Agilent 1 to Lumencor Green TTL input
        d. Agilent, Keith, attached to GPIB cord
    3. Run Imports Cell
    4. Change "user inputs" section as desired
    5. Run user inputs cell
    6. Run: light = lumencor()
    7. Get everything focused with camera and micromanager program
    8. Run Program
'''

##############################################################################
# USER INPUT REGION
##############################################################################

# 1. The IDS: -----------------------------------------------------
classID = '210412_Swift_Devices'
expt_str = 'Swft_Device_MA4_1sun_25C_50RH_air'
sampleID = '210519MA4'

#2. Sun power settings: -------------------------------------------
one_sun_pwr = 129 # For Lumencor, 255 scale; 129 = 1 sun with Olympus 5X BD objective and ND25 filter

#3. Measurement time: ---------------------------------------------
measurement_time = 15*60 # s, how much time to allocate for total PLVA-JV-R DAQ cycle

#4. The cycle int to start from when run is interrupted and should be continued from an intermediate cycle
measurement_start = 70



##############################################################################
# Other settings to be rarely changed.
##############################################################################
import numpy as np
import os

# Lumencor and waveform generator settings
shape = 'SQU'
frequency_low = 10 # Hz
frequency_high = 400 # Hz
amplitude = 5 # V

# Keithley Settings
start_v = 1.2 # V
jv_step_num = 25 # V
appl_v = np.linspace(start_v,0, num=jv_step_num) # V
const_V = 0.75
Vmpp = 0.75 # provide initial guess for maximum power point, V
Vmpp_scanrange = 0.1 # voltage range to scan over to find MPP (+/-)
Vmpp_scanstep = 0.01 # step size for MPP scan
Vmpp_nsteps = int(2*Vmpp_scanrange/Vmpp_scanstep + 1) # calculate how many steps for short MPP scan

# Experimental details
n_gradients = 1 # number of gradients to measure
n_gradpoints = 1 # number of points to measure along the gradient
n_repeats = 1000 # number of times to collect data at the same point - same as number of primary videos in the corresponding PLVA analysis
n_measurements = n_gradients*n_gradpoints*n_repeats # total number of locations * number of different intensity levels * time intervals
t_step = 0.2 # s, time interval to collect data

plva_wait_time = 15 #s
check_time = 2 # for checking the temp folders
#is_autofocus=True #are we using autofocus?
#if is_autofocus:
#    plva_wait_time = plva_wait_time + 15 #s
t_PL = plva_wait_time
motor_move_time = 15 # time for filter cube wheel to complete 1/4 revolution, s
DF_DAQ_time = 5 # time allocated for dark field image acquisition, s

t_MPP_scan = t_step*Vmpp_nsteps # time to search for MPP
t_MPP_soak = 10 # time to collect at MPP
t_MPP = t_MPP_scan + t_MPP_soak # total time for MPPT
t_Isc = 10 # time for steady Isc measurements

jv_time = 20
jv_alloc = 30
voc_time = 60 #s

directory = os.path.join("C:\\", "Users", "Administrator", "Documents",
                         "Users", "Wiley", "PLVA Measurements")
directory = os.path.join(directory, classID, expt_str)
save_name = os.path.join(directory, "electronic_measurement_data")

# assign which types of electrical measurements to make - sync this up with Beanshell IN ORDER
meas_to_make = {
                'Dark_Field' : True, # move motor so beanshell can snap an image
                'Voc_steady' : True, # sample Voc while beanshell takes PL video
                'MPP' : True, # set MPP and sample(beanshell idle)
                'Isc_steady' : True, # set V = 0 and measure Isc (beanshell idle)
                'Light_IV' : True, # light IV sweep (beanshell idle)
                'Dark_IV' : True, # dark IV sweep (beanshell idle)
                'Bright_Field' : False, # DO NOT CHANGE - Bright field not set up yet.
               }

# Enter starting guesses for PL, DF position
z_PL = 1905
z_DF = 2859

# End of user input section ##################################################

#%%
# Imports
from pymeasure.instruments.agilent import Agilent33220A
from pymeasure.instruments.keithley import Keithley2400

from time import sleep, time
import serial
import pandas as pd
import datetime
import sys
import subprocess
import re
import json
from glob import glob

sys.path.append(r'C:\Users\Administrator\Desktop\bs2py\microscope_1\motor')
from motor_update_slider import Stepper_motor_slider

# comment the line below for running this section alone with shift+Enter
sys.path.append(r'C:\Users\Administrator\Desktop\bs2py')
from countdown import countdown

# Connect to GPIB instriments using pymeasure modules
wave_gen = Agilent33220A("GPIB::10")
keithley = Keithley2400("GPIB::24")
#keithley_too = Keithley2400("GPIB::20") 

# Define lumencor class for all light control functions
class lumencor(object):
    '''
    Lumencor class
    '''
    
    def __init__(self):
        try:
            self.M = serial.Serial('COM3',9600)
            self.M.write(bytes.fromhex('57 02 FF 50'))
            self.M.write(bytes.fromhex('57 03 AB 50'))
            self.M.write(bytes.fromhex('4F 7F 50'))
        except serial.SerialException:
            print('Lumencor already initialized')

    def set_intensity(self, pwr):
        intensity = round((1 - pwr/255)*255)    
        tmp = hex(intensity)   
        str1 = 'f' + tmp[2]
        if len(tmp) > 3:
            str2 = tmp[3] + '0'
        else:
            str2 = '00'    
        self.M.write(bytes.fromhex('53 18 03 04 ' + str1 + ' ' + str2 + ' 50'))
        self.pwr = pwr
        
    def strobe(self, shape, freq, amp):
        wave_gen.shape = shape
        wave_gen.frequency = freq
        wave_gen.amplitude = amp
        wave_gen.offset = 0
        wave_gen.output = True
        
    def steady(self):
        wave_gen.amplitude = .01
        wave_gen.offset = 4
        wave_gen.output = True
        
    def off(self):
        wave_gen.output = False

#%%
#light = lumencor()
#motor = Stepper_motor_slider()

#%% ##############################################################################
# determine how many points to add to the electronic measurement data file for each DAQ cycle
# and how much time to allocate for all the measurements
t_alloc = 0
n_DAQpoints = 0
if meas_to_make['Dark_Field']:
    t_alloc += 2*motor_move_time + DF_DAQ_time
if meas_to_make['Voc_steady']:
    n_DAQpoints += t_PL/t_step 
    t_alloc += t_PL
if meas_to_make['MPP']:
    n_DAQpoints += t_MPP/t_step 
    t_alloc += t_MPP
if meas_to_make['Isc_steady']:
    n_DAQpoints += t_Isc/t_step
    t_alloc += t_Isc
if meas_to_make['Light_IV']:
    n_DAQpoints += 2*jv_step_num 
    t_alloc += (2*jv_step_num)*t_step
if meas_to_make['Dark_IV']:
    n_DAQpoints += 2*jv_step_num
    t_alloc += (2*jv_step_num)*t_step 
n_DAQpoints = int(n_DAQpoints)

# ###########################################################################
#defining the paths related to MM files for PL acquisition--------------------
CONFIG_FILE = "C:/Users/Administrator/Desktop/bs2py/microscope_1/Photoconductivity2.cfg"
PLAQ_CODE = "C:/Users/Administrator/Desktop/bs2py/plaq.py"

DF = 1 if meas_to_make['Dark_Field'] else 0
BF = 1 if meas_to_make['Bright_Field'] else 0
INIT_POINTS="C:/Users/Administrator/Desktop/bs2py/points.npy" # A dummy variable not used when XY stage doesn't move.

args = [CONFIG_FILE, directory, str(n_measurements), str(n_gradients), str(n_gradpoints),
        INIT_POINTS, str(measurement_start),
        str(DF), str(BF), sampleID, str(z_PL), str(z_DF)]

# The PL acquisition occurs only if it finds the temporary folder
def plaq_process():
    return subprocess.Popen(['python27', PLAQ_CODE, *args])
#-----------------------------------------------------------------------------
# ###########################################################################
#%%
# Configure keithley to desired settings (shouldn't need to change these)
def keithley_applyV_measJ():
    keithley.apply_voltage()                # Sets up to source voltage
    keithley.source_voltage_range = 10      # Sets the source voltage range to 10 V
    keithley.compliance_current = .1        # Sets the compliance current to .10 A
    #keithley.source_voltage = 0             # Sets the source Voltage to 0 V
    keithley.enable_source()                # Enables the source output
    keithley.measure_current()              # Sets up to measure current
def keithley_applyJ_measV():
    keithley.apply_current()                # Sets up to source voltage
    keithley.source_current_range = .1      # Sets the source voltage range to 100 ,A
    keithley.compliance_voltage = 21     # Sets the compliance voltage to 21 V
    keithley.source_current = 0             # Sets the source current to 0 A
    keithley.enable_source()                # Enables the source output
    keithley.measure_voltage()              # Sets up to measure voltage
'''
keithley_too.apply_voltage()                # Sets up to source current
keithley_too.source_voltage_range = 0      # Sets the source voltage range to 10 V
keithley_too.compliance_current = 1e-2        # Sets the compliance current to .10 A
keithley_too.source_voltage = 0             # Sets the source Voltage to 0 V
keithley_too.enable_source()                # Enables the source output
keithley_too.measure_current()            # Sets up to measure voltage
'''


# initialize light
#light = lumencor()
light.set_intensity(one_sun_pwr)
light.steady()

# Pre-allocate data matrix
n_ts = int(jv_time/t_step)
ts = np.linspace(0, measurement_time-t_step, num=n_ts)
data = np.zeros([n_DAQpoints, 9]) 

# identify turning points in the jv sweep
idx_jv_start = 0
idx_jv_mid = idx_jv_start + jv_step_num
idx_jv_mid_jsc = idx_jv_mid
idx_jv_end = idx_jv_start + 2*jv_step_num + 2/t_step
# cols: [primary video# gradient# position# meas_t keithley_i srs_i keithley_v suns_setting]
# light_change_idx = int(n_ts*stab_perc)
# pwr_idx = 0

# numerical codes to describe the type of measurement
status_code = {
               'Dark_Field' : 0, # move motor so beanshell can snap an image
               'Voc_steady' : 1, # sample Voc while beanshell takes PL video
               'MPP'        : 2, # set MPP and sample(beanshell idle)
               'Isc_steady' : 3, # set V = 0 and measure Isc (beanshell idle)
               'Light_IV'   : 4, # light IV sweep (beanshell idle)
               'Dark_IV'    : 5, # dark IV sweep (beanshell idle)
              }
        
time0_values = {
    'Voc_steady' : [], # sample Voc while beanshell takes PL video
    'MPP'        : [], # set MPP and sample(beanshell idle)
    'Isc_steady' : [], # set V = 0 and measure Isc (beanshell idle)
    }

if measurement_start != 0:
    time0_data = pd.read_csv(save_name + '_cycle0.csv')
    status = time0_data.values[:, 8]
    
    for key in time0_values:
        col = 6
        if key == 'Isc_steady':
            col = 4
        time0_values[key] = np.median(time0_data[status==status_code[key]].values[:, col])

##################################################################################{
#"""
# START OF THE EXPERIMENT : ---------------------------------------------
if not os.path.exists(os.path.join(directory, 'temp')):
    #Make the temporary directory to initiate PL acquisition
    os.makedirs(os.path.join(directory, 'temp'))

#starts the plaq code in the background
plaq = plaq_process()
#"""
##################################################################################}

# Main loop where data is collected - ii increments every DAQ cycle
for ii in range(measurement_start, n_measurements, 1):
    t = time() # make note of the beginning of the DAQ cycle
    print('Cycle: ',ii)
    
    keithley.apply_current()
    keithley_applyJ_measV()
    
    time_ii_norm_values = {
        'Voc_steady' : [], # sample Voc while beanshell takes PL video
        'MPP'        : [], # set MPP and sample(beanshell idle)
        'Isc_steady' : [], # set V = 0 and measure Isc (beanshell idle)
    }
    
    # prior to collecting electronic data, do dark field, if this is an option; skip otherwise
    if meas_to_make['Dark_Field']:
        while not os.path.exists(os.path.join(directory, 'DF_temp')):
            sleep(int(check_time/2))
        print('Status: Collecting Dark Field...')
        
        motor.run(2) # move cube to DF position
        #sleep(motor_time)
        os.rmdir(os.path.join(directory, 'DF_temp')) # This is the cue to plaq to take DF image
        print('Cycle {} : Status: Acquiring DF...'.format(ii))
              
        #wait till DF image is taken
        while not os.path.exists(os.path.join(directory, 'DF_temp')):
            sleep(int(check_time/2))
            
        motor.run(3) # return cube to PL position
        os.rmdir(os.path.join(directory, 'DF_temp'))
    
    print('\n\nCycle {} : Status: Acquiring PLVA...'.format(ii))

    ##############################################################################{
    #"""
    while os.path.exists(os.path.join(directory, 'temp')):
        sleep(check_time/2)
    
    #Reading metadata
    vidFolder = glob(os.path.join(directory, "*_grad0_loc0_time" + str(ii)))[0]
    
    # Extended path length
    vidFolder = u'\\'.join(re.split('[\\\\/]', vidFolder))
    if not vidFolder.startswith('\\\\?\\'):
        vidFolder = u'\\\\?\\'+vidFolder

    with open(os.path.join(vidFolder, 'MMStack_Pos0_metadata.txt'), 'r') as file:
        meta_dict = json.load(file)
    
    print('Exposure Time : ' + str(meta_dict['FrameKey-0-0-0']['HamamatsuHam_DCAM-Exposure']) + ' ms')
    #"""
    ##############################################################################
    
    #n_ts = int(jv_time/t_step)
    # initialize list of times within a given DAQ cycle
    ts = np.linspace(0, t_alloc-t_step, num=n_DAQpoints)
    
    n_idx = 0 # which datapoint are we on for a given cycle

    # do PL/Voc measurements
    if meas_to_make['Voc_steady']:
        print('Status: Collecting PL/Voc...')
        light.steady() # turn on light
        # set V = 0 and set up to measure voltage
        keithley.apply_current()
        keithley_applyJ_measV()
        # timestamp the start of Voc collection to register against the PL video
        present = present=datetime.datetime.now()
        timestring = str(present.year) + '-' + str(present.month) + '-' + str(present.day) + '_' + str(present.hour) + ':' + str(present.minute) + ':' + str(present.second) + '.' + str(present.microsecond)
        df=pd.DataFrame(data={'Timestamp',timestring}) 
        df.to_csv(os.path.join(directory, 'Timestamp_cycle'+str(ii)+'.csv'))
        #np.savetxt(directory+'Timestamp_cycle'+str(ii)+'.csv', df, delimiter=',')
        
        # log data (jj increments over timesteps)
        for jj in range(int(t_PL/t_step)):
            t_this = time() # make note of the beginning of timestep
            
            # populate the data time series
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.voltage
            if ii == 0:
                time0_values['Voc_steady'].append(data[n_idx, 6])
            else:
                time_ii_norm_values['Voc_steady'].append(data[n_idx, 6])
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['Voc_steady'] # encode the type of measurement            
            n_idx += 1 # and increment the counter
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0]))    

    # do maximum power point measurements
    if meas_to_make['MPP']:
        print('Status: Maximum power point tracking...')
        light.steady() # turn on light
        # set up to measure voltage
        keithley.apply_voltage()
        keithley_applyV_measJ()
        # determine what voltage range to scan based on previous MPP
        Vs_to_scan = np.linspace(Vmpp-Vmpp_scanrange,Vmpp+Vmpp_scanrange,num=Vmpp_nsteps)
        # initialize current measurements
        I_mppscan = np.zeros([Vmpp_nsteps])
        
        # first do a short scan over the near-MPP range
        for jj in range(Vmpp_nsteps):
            
            t_this = time() # make note of the beginning of timestep
            
            # step the voltage and measure the current over the short range
            keithley.ramp_to_voltage(Vs_to_scan[jj], steps=2, pause=.02)
            I_mppscan[jj] = keithley.current
            
            
            # log the data
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.source_voltage
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['MPP'] # encode the type of measurement 
            n_idx += 1 # and increment the counter
            
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0])) 
        
        # after scanning, redetermine Vmpp
        # calculate the power
        power = I_mppscan*Vs_to_scan
        # reassign the corresponding MPP voltage
        Vmpp = Vs_to_scan[np.argmin(power)]
        
        
        # now soak at MPP... 
        keithley.ramp_to_voltage(Vmpp, steps=2, pause=.02)
        # ...and log data
        for jj in range(int(t_MPP_soak/t_step)):
            t_this = time() # make note of the beginning of timestep
            

            # populate the data time series
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.source_voltage
            if ii == 0:
                time0_values['MPP'].append(data[n_idx, 6])
            else:
                time_ii_norm_values['MPP'].append(data[n_idx, 6])
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['MPP'] # encode the type of measurement            
            n_idx += 1 # and increment the counter
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0])) 

    # do steady Isc measurements
    if meas_to_make['Isc_steady']:
        print('Status: Collecting Isc...')
        light.steady() # turn on light
        # set V = 0 and set up to measure current
        keithley.apply_voltage()
        keithley_applyV_measJ()
        keithley.ramp_to_voltage(0, steps=2, pause=.02)
        
        # log data (jj increments over timesteps)
        for jj in range(int(t_Isc/t_step)):
            t_this = time() # make note of the beginning of timestep
            
            # populate the data time series
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            if ii == 0:
                time0_values['Isc_steady'].append(data[n_idx, 4])
            else:
                time_ii_norm_values['Isc_steady'].append(data[n_idx, 4])
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.voltage
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['Isc_steady'] # encode the type of measurement
            n_idx += 1 # and increment the counter
            
            
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0]))   
            
    # do light I-V sweep
    if meas_to_make['Light_IV']:
        print('Status: Collecting light I-V...')
        light.steady() # turn on light
        # set V = 0 and set up to measure current
        keithley.apply_voltage()
        keithley_applyV_measJ()
        
        # log data (jj increments over timesteps)
        for jj in range(int(2*jv_step_num)):
            t_this = time() # make note of the beginning of timestep
            
            # at start, voltage index = 0
            if jj == idx_jv_start:
                v_idx = 0
           
            # reverse sweep: ramp voltage down
            if (jj >= idx_jv_start) and (jj < idx_jv_mid):
                keithley.apply_voltage()
                keithley.ramp_to_voltage(appl_v[v_idx], steps=2, pause=.02)
                v_idx += 1
           
            # forward sweep: ramp voltage up    
            if (jj >= idx_jv_mid_jsc) and (jj < idx_jv_end):
                v_idx -= 1
                keithley.apply_voltage()
                keithley.ramp_to_voltage(appl_v[v_idx], steps=2, pause=.02)
            
            
            # populate the data time series
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.source_voltage
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['Light_IV'] # encode the type of measurement
            n_idx += 1 # and increment the counter
            
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0]))    

    # do dark I-V sweep
    if meas_to_make['Dark_IV']:
        print('Status: Collecting dark I-V...')
        light.off() # turn light off
        # set V = 0 and set up to measure current
        keithley.apply_voltage()
        keithley_applyV_measJ()
        
        # log data (jj increments over timesteps)
        for jj in range(int(2*jv_step_num)):
            t_this = time() # make note of the beginning of timestep
            
            # at start, voltage index = 0
            if jj == idx_jv_start:
                v_idx = 0
           
            # reverse sweep: ramp voltage down
            if (jj >= idx_jv_start) and (jj < idx_jv_mid):
                keithley.apply_voltage()
                keithley.ramp_to_voltage(appl_v[v_idx], steps=2, pause=.02)
                v_idx += 1
           
            # forward sweep: ramp voltage up    
            if (jj >= idx_jv_mid_jsc) and (jj < idx_jv_end):
                v_idx -= 1
                keithley.apply_voltage()
                keithley.ramp_to_voltage(appl_v[v_idx], steps=2, pause=.02)
            
            
            # populate the data time series
            data[n_idx, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
            pos_number = ii % n_gradpoints # 0
            data[n_idx, 1] = 0 # gradient index
            data[n_idx, 2] = pos_number # position on gradient index
            data[n_idx, 3] = ts[n_idx] #this measurement time
            data[n_idx, 4] = keithley.current # keithley measured current
            data[n_idx, 5] = 0 #srs output, converted to current
            data[n_idx, 6] = keithley.source_voltage
            data[n_idx, 7] = one_sun_pwr # beam intensity
            data[n_idx, 8] = status_code['Dark_IV'] # encode the type of measurement
            n_idx += 1 # and increment the counter
            
            # wait for remainder of the timestep
            t_this2 = time()
            sleep(np.max([t_step - (t_this2-t_this),0])) 
    
    # after conclusion of measurement modules, turn light back on...
    light.steady()
    # ...and return to MPP
    keithley.apply_voltage()
    keithley_applyV_measJ()
    keithley.ramp_to_voltage(Vmpp, steps=2, pause=.02)
    # save the electronic data  
    np.savetxt(save_name + '_cycle'+str(ii)+'.csv', data, delimiter=',')
    
    if ii == 0:
        time0_values = {key: np.median(time0_values[key]) for key in time0_values}
        time_ii_norm_values = time0_values.copy()
    else:
        time_ii_norm_values = {key: np.median(time_ii_norm_values[key]) for key in time_ii_norm_values}
    
    print("Status: Jsc-{:.1f}, Voc-{:.1f}, Vmpp-{:.1f} ... ".format(
            100*time_ii_norm_values['Isc_steady']/time0_values['Isc_steady'],
            100*time_ii_norm_values['Voc_steady']/time0_values['Voc_steady'],
            100*time_ii_norm_values['MPP']/time0_values['MPP'],
        ))
    
    # wait for measurement to finish
    elapsed = time()-t
    print('Status: Waiting for next data acquisition cycle...')
    #sleep(measurement_time - elapsed)
    
    ###########################################################################
    countdown(measurement_time - elapsed)
    os.mkdir(os.path.join(directory, 'temp'))
    ###########################################################################

# At the end of the experiment, turn equipment off        
keithley.shutdown()  
light.off()                

print('Status: Complete.')



