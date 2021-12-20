# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:39:58 2020
@author: Ryan Stoddard
@Modified: Preetham
"""
'''
Procedure:
    1. Turn on Lumencor, Agilent, keithley, SRS
    2. Check connections
        a.Agilent sync to SRS
        b. Keithley V thru the SRS (like in EQE Measurement)
        c. Agilent 1 to Lumencor Green TTL input
        d. Agilent, Keith, SRS attached to GPIB cord
    3. Run Imports Cell
    4. Change "user inputs" section as desired
    5. Run user inputs cell
    6. Run: light = lumencor()
    7. Get everything focused with camera and micromanager program
    8. Run Program
'''

####################################################################
# USER INPUT REGION
###################################################################

# 1. The IDS: -----------------------------------------------------

"""classID = '212308_Gradients_Evan'
material_str = ''
expt_str = 'PL_PC_T_FA80Cs20Br17I81SCN020_FA755Cs195PEA050Br17I81SCN020_075MACL_25C_40RH_4'"""


"""
classID = '210418_High_Eg_Films'
material_str = '(Cs, FA)Pb(I, Br)3'
expt_str = 'PL_DF_PC_T_1sun_65C_50RH_air'
"""

""
classID = '211214_Swift_Samples'
material_str = ''
expt_str = 'PL_PC_T_DF_FA80Cs20PbI90Br10_1sun_25C_50RH_Air_4'


measurement_type = '2-point' # indicate the contact topology


# 2. The sun intensities :-----------------------------------------
"""
=================================================
|     100x lens Sun intensities in 255 scale    |
|-----------------------------------------------|
|      ND6 filter IN      ||   ND6 filter OUT   |
|-------------------------||--------------------|
| Sun Power |     Value   ||  Sun Power | Value |
|-------------------------||--------------------|
|     1     |   15-->21   ||      16    |  14   | 14
|     2     |   27        ||      32    |  24   |
|     4     |   52        ||      64    |  45   |
|     8     |  106-->167  ||     128    |  90   | 101
======================================================
|        50x Olympus PL-DF lens (TENTATIVE)          |
|----------------------------------------------------|
|                    ND6 filter IN                   |
|----------------------------------------------------|
| Sun Power | Excel Calc. Value | Probe comp. value  |
|----------------------------------------------------|
|     1     |        12         |         10         |
|     2     |        21         |         16         |
|     4     |        40         |         27         |
|     8     |        78         |         55         |
======================================================
=================================
|  50x Olympus PL-DF LWD lens   |
|-------------------------------|
|        ND25 filter IN         |
|-------------------------------|
| Sun Power |     Calc. Value   |
|-------------------------------|
|     1     |        13         | 12 (current value)
|     8     |        80         | 74 (current value) 
=================================

# For Lumencor, 255 scale; 1 Sun = 15 for 100X with ***ND6 filter*** in, 71 for 5X with ***ND25 filter***
"""
probe_power = 74 # <------- illumination intensity for MEASUREMENTS (before 235 - 4pt probe)
soak_power = 12 #<------- illumination intensity for STRESS (before 46 - 4pt probe)

# settings for lock-in: sometimes edit
time_constant = .1 # s
#shunt_R = 10316000 # ohms
shunt_R = 0.464e6 #ohms - this setting is good for single perovskite samples
#shunt_R = 14.67e3 #ohms - this setting is good for perovskite gradients

# 3. The cycle time : ---------------------------------------------
# how much time to allocate for total PLVA/LD DAQ cycle aka cycle period
measurement_time = 5*60     # s  ----> This is time between two locations

# FOR SPIN-COATED SAMPLES, use only the above variable and 
# keep meaasurement_time2 = None

#measurement_time2 = 60  # s  ----> This is the time between two cycles
measurement_time2 = None  # s  ----> This is the time between two cycles

# 4. Sensitivity : ------------------------------------------------
# settings for lock-in: always edit
sensitivity = 100e-3 # V 
                        # 100 mV used before for 4pt probe
                       # 10 mV so far good for perovskite gradients at 16 suns, 15 kOhm shunt resistor
                       # 20 mV so far good for perovskite single samples at 8 suns, 464 kOhm resistor
                       # 50 mV good for perovskite single samples at 32 suns with 464 kOhm resistor
                       #
                       # use 1e-3 for ultrathin samples
                       # use 50 mV for very thick samples
                     

# 5. Changes to the path if required
if material_str == '':
    save_name = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\'+ expt_str
else:
    save_name = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\' + material_str + '\\' + expt_str


####---####---####---####---####---####---####---####---####---####---
# 6. Data to enter only for GRADIENTS :
n_gradients = 1 # number of gradients to measure
n_gradpoints = 1 # number of points to measure along the gradient

# x, y, z cordinates of anchors
points = [
      #  [1374, -4713, 7248],     #---> starting composition co-ordinates
        #[15851, 11424, 276],
       # [-5229, -4732, 7339],     #---> middle composition co-ordinates
        #[8789, 11009, 495],
        #[-11832, -4778, 7311],      #---> ending composition co-ordinates
        
]

####---####---####---####---####---####---####---####---####---####---


# 7. Number of cycles : ------------------------------------------
# number of times to collect data at the same point
# same as number of primary videos in the corresponding PLVA analysis
#DONT CHANGE THIS - THIS WILL CAUSE ERROR IN DATA PIPELINE
n_repeats = 400


# 8. Set this to True if you want dark field
# NOTE : Note that this dark field code works only when the code is run WITHOUT micromanager
dark_field = True
bright_field = False

# 9. Use this to start an interrupted run from the <measurement_start>th cycle
measurement_start = 0

switch_polarity = True
####################################################################
# END OF USER INPUT REGION
####################################################################

"""
Below this point, all the snippets of code enclosed within ###### lines
are additions by Preetham. They can be removed to revert to the original
version.
"""

#%%
# Imports
from pymeasure.instruments.agilent import Agilent33220A
from pymeasure.instruments.srs import SR830
from pymeasure.instruments.keithley import Keithley2400
import numpy as np
import sys
from time import sleep, time
import subprocess
import os
import re
import json
import sys
sys.path.append('.')
sys.path.append('../')
import serial
import pandas as pd

sys.path.append(r'C:\Users\Administrator\Desktop\bs2py\microscope_1\motor')
from motor_update_slider import Stepper_motor_slider

# comment the line below for running this section alone with shift+Enter
sys.path.append(r'C:\Users\Administrator\Desktop\bs2py')
from countdown import countdown


# Connect to GPIB instriments using pymeasure modules
wave_gen = Agilent33220A("GPIB::10")
srs = SR830("GPIB::8")
keithley = Keithley2400("GPIB::24")
keithley_too = Keithley2400("GPIB::20") 

# Define lumencor class for all light control functions
class lumencor(object):
    '''
    Lumencor class

    '''
    
    def __init__(self):
        try:
            self.M = serial.Serial('COM3',baudrate=9600)
            self.M.write(bytes.fromhex('57 02 FF 50')) # config statement 1
            self.M.write(bytes.fromhex('57 03 AB 50')) # config statement 2
            # the above two statements are necessary to configure the Lumencor
            # for TTL communication
            self.M.write(bytes.fromhex('4F 7F 50')) # disables all lights (everything off initially)
        except serial.SerialException:
            print('Lumencor already initialized *or* something else is trying to talk over COM3')

    def set_intensity(self, pwr):
        # set up the command for the green light
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
        
#######################################################################
# COMMENT THIS LINE OFF IF LUMENCOR RELATED ERROR OCCURS
#light = lumencor()
#motor = Stepper_motor_slider()
#######################################################################


#%%
# Less frquently edited variables and experimental settings ------------------

# settings for lock-in: never edit
Vpkpk_per_Vrms = 2.23 #For square wave, also accounting fo the fact that lock-in amplifier measures only first sinusoidal harmonic of the input square wave signal

# Reference to sun power
# Lumencor and waveform generator settings: never edit
# 1,2,4,8,16,32,64,128 suns for Lumencor 255 scale (note filter change)
suns_pwr_settings = np.array([15,27,52,106,14,24,45,90]) 

#Lumencor settings
shape = 'SQU'
frequency_low = 10 # Hz
frequency_high = 400 # Hz
amplitude = 5 # V

# Keithley Settings: never edit
appl_v = 3 # V - for 2 point measurements
appl_I = 10e-9 # A - for 4-point measurements
#appl_v = np.linspace(-3 ,3, num=7) # V

#Measurement csv path
csv_save_name = os.path.join(save_name, 'electronic_measurement_data.csv')

##################################################################
#Saving the anchors only for gradients
INIT_POINTS="C:/Users/Administrator/Desktop/bs2py/points.npy"
points = np.array(points)
if not (n_gradpoints == 1 and n_gradients == 1):
    np.save(INIT_POINTS, points)
###################################################################
    
n_measurements = n_gradients*n_gradpoints*n_repeats # total number of locations * number of different intensity levels * time intervals

check_time = 2 # s, time taken between two consecutive checks for temp folder by electronic measuring code
plva_time = 20 # s, during this time light is steady and keithley is off
if measurement_type == '4-point':
    ld_time = 32
    switch_t1 = 8.0
    switch_t2 = 16.0
    switch_t3 = 24.0
else:   
    ld_time = 16 # s, total time, including stabilization and measurement
    switch_t1 = 4.0 # time to switch from dark to light
    switch_t2 = 8.0 # time to switch from light to low freq strobe
    switch_t3 = 12.0 # time to switch from low to high freq strobe
stab_perc = 0.3 # fraction of ld_time that light is off for "zero"
t_step = 0.2 # s, time interval to collect data
motor_time = 8 #s, time it takes for motor to switch from one mode to another
DF_time = 8 #s, time it takes to take DF image and for the led to stay turned on
BF_time = 8 #s, time it takes to take DF image and for the led to stay turned on


# End of settings section----------------------------------------------------

# ###########################################################################
#defining the paths related to MM files for PL acquisition--------------------
CONFIG_FILE = "C:/Users/Administrator/Desktop/bs2py/microscope_1/Photoconductivity2.cfg"
PLAQ_CODE = "C:/Users/Administrator/Desktop/bs2py/plaq.py"

DF = 1 if dark_field else 0
BF = 1 if bright_field else 0

args = [CONFIG_FILE, save_name, str(n_measurements), str(n_gradients), str(n_gradpoints),
        INIT_POINTS, str(n_gradients*n_gradpoints*measurement_start),
        str(DF), str(BF), '0']



# The PL acquisition occurs only if it finds the temporary folder
def plaq_process():
    return subprocess.Popen(['python27', PLAQ_CODE, *args])
#-----------------------------------------------------------------------------
# ###########################################################################


#%%
# Configure keithley to desired settings (shouldn't need to change these)

if measurement_type == '4-point':
    keithley.apply_current(compliance_voltage=5) # set compliance voltage to be relatively low
    keithley.source_current_range = 15e-9
    keithley.source_current = 10e-9
    keithley.wires = 4 # set up to measure in 4-pt mode
    keithley.enable_source()
    keithley.measure_voltage()
else:
    keithley.apply_voltage()                # Sets up to source voltage
    keithley.source_voltage_range = 10      # Sets the source voltage range to 10 V
    keithley.compliance_current = .1        # Sets the compliance current to .10 A
    keithley.source_voltage = 0             # Sets the source Voltage to 0 V
    keithley.enable_source()                # Enables the source output
    keithley.measure_current()              # Sets up to measure current


keithley_too.apply_voltage()                # Sets up to source current
keithley_too.source_voltage_range = 0      # Sets the source voltage range to 10 V
keithley_too.compliance_current = 1e-2        # Sets the compliance current to .10 A
keithley_too.source_voltage = 0             # Sets the source Voltage to 0 V
keithley_too.enable_source()                # Enables the source output
keithley_too.measure_current()            # Sets up to measure voltage

# set srs settings
srs.sensitivity = sensitivity
srs.time_constant = time_constant
V_full_scale = 1 # V

# initialize light
#light = lumencor()
light.set_intensity(probe_power)
light.steady()

# Pre-allocate data matrix
n_ts = int(ld_time/t_step)
ts = np.linspace(0, ld_time-t_step, num=n_ts)
if measurement_start>0:
    data = np.genfromtxt(csv_save_name, delimiter=',')
else:
    data = np.zeros([n_measurements*n_ts, 10])
# cols: [primary video# gradient# position# meas_t keithley_i srs_i keithley_v suns_setting]
# light_change_idx = int(n_ts*stab_perc)
# pwr_idx = 0

##################################################################################{
#"""
# START OF THE EXPERIMENT : ---------------------------------------------
if not os.path.exists(os.path.join(save_name, 'temp')):
    #Make the temporary directory to initiate PL acquisition
    os.makedirs(os.path.join(save_name, 'temp'))

#starts the plaq code in the background
plaq = plaq_process()
#"""
##################################################################################}

# Main loop where data is collected
for ii in range(n_gradients*n_gradpoints*(measurement_start), n_measurements, 1):
    
    grad_index, pos_number = divmod(ii, n_gradpoints) # 0s for spin coated samples
    time_count, grad_index = divmod(grad_index, n_gradients)
    
    t = time() # make note of the beginning of acquisition
    light.set_intensity(probe_power)
    light.steady()

    
    if dark_field:
        while not os.path.exists(os.path.join(save_name, 'DF_temp')):
            sleep(int(check_time/2))
        motor.run(2)
        sleep(motor_time)
        os.rmdir(os.path.join(save_name, 'DF_temp'))
        
        print('Cycle {} | Grad {} | Loc {} : Status: Acquiring DF...'.format(time_count, grad_index, pos_number))
        
        #wait till DF image is taken
        while not os.path.exists(os.path.join(save_name, 'DF_temp')):
            sleep(int(check_time/2))
        
        if not bright_field:
            #Change back to PL
            motor.run(3)
            #############################
            sleep(motor_time)
        os.rmdir(os.path.join(save_name, 'DF_temp'))
    
    # next take bright field image, if desired    
    if bright_field:
        while not os.path.exists(os.path.join(save_name, 'BF_temp')):
            sleep(int(check_time/2))
        motor.run(1)
        sleep(motor_time)
        os.rmdir(os.path.join(save_name, 'BF_temp'))
        
        print('Cycle {} | Grad {} | Loc {} : Status: Acquiring BF...'.format(time_count, grad_index, pos_number))
        
        #wait till DF image is taken
        while not os.path.exists(os.path.join(save_name, 'BF_temp')):
            sleep(int(check_time/2))
        
        # Return filter cube to PL position
        # do this in 2 parts so the motor "retraces its steps"
        motor.run(2)
        sleep(motor_time-2) 
        motor.run(3)
        sleep(motor_time)
        os.rmdir(os.path.join(save_name, 'BF_temp'))



    print('\n\nCycle {} | Grad {} | Loc {} : Status: Acquiring PLVA...'.format(time_count, grad_index, pos_number))

    ##############################################################################{
    #"""
    while os.path.exists(os.path.join(save_name, 'temp')):
        sleep(check_time/2)
    
    #Reading metadata
    vidName = re.split('[\\\\/]', save_name)[-1] + "_grad"+ str(grad_index) + "_loc"+ str(pos_number) +"_time" + str(time_count)
    vidFolder = os.path.join(save_name, vidName)

    # Extended path length
    vidFolder = u'\\'.join(re.split('[\\\\/]', vidFolder))
    if not vidFolder.startswith('\\\\?\\'):
        vidFolder = u'\\\\?\\'+vidFolder

    with open(os.path.join(vidFolder, 'MMStack_Pos0_metadata.txt'), 'r') as file:
        meta_dict = json.load(file)
    
    print('Cycle {} | Grad {} | Loc {} : Exposure Time : '.format(time_count, grad_index, pos_number) + str(meta_dict['FrameKey-0-0-0']['HamamatsuHam_DCAM-Exposure']) + ' ms')
    #"""
    ##############################################################################}
    #UNCOMMENT line below for original version
    #sleep(plva_time) # wait for video to be acquired
    ###############################################################################
    

    light.set_intensity(0) # turn off the light
    #light.set_intensity(probe_power)
    
    # configure the light strobe profile (waveform, frequency, amplitude), starting with low frequency
    #light.strobe(shape, frequency_low, amplitude)
    
    # Bump up sensitivity to keep pace every time the lamp intensity increases by an order of magnitude
#    if suns_pwr_settings[pwr_idx] == 14 or suns_pwr_settings[pwr_idx] == 90:
#        srs.sensitivity = sensitivity*10
    

    if measurement_type == '4-point':
        appl_I = -appl_I
        # Apply the measurement bias:
        keithley.ramp_to_current(appl_I, steps=2, pause=.02)
    else:
        # Flip the sign applied voltage every DAQ cycle to avoid charge buildup due to ion migration
        # Comment the line below out if you don't want to do this
        if switch_polarity:
            appl_v = -appl_v
        
        # Apply the measurement bias:
        keithley.ramp_to_voltage(appl_v, steps=2, pause=.02)
        #keithley.ramp_to_voltage(appl_v[0], steps=2, pause=.02)
        #v_idx = 0 # initialize voltage change index (for IV sweep only)
    
    condition = float('nan')
    print('Cycle {} | Grad {} | Loc {} : Status: Acquiring dark current...'.format(time_count, grad_index, pos_number))
    
    # this loop controls data collection
    for jj in range(n_ts):
        
        # After 5 seconds have passed, turn on the lamp for DC measurements
        if jj*t_step == switch_t1:
            light.set_intensity(probe_power)
            light.steady()
            condition = 0
            print('Cycle {} | Grad {} | Loc {} : Status: Acquiring DC light current...'.format(time_count, grad_index, pos_number))
            #keithley_too.ramp_to_current(0)             # Ramps the current to 0 mA
            #print(keithley_too.voltage) 
        # After 5 seconds have passed, turn on the low-frequency strobe
        if jj*t_step == switch_t2:
            light.strobe(shape, frequency_low, amplitude)
            condition = frequency_low
            print('Cycle {} | Grad {} | Loc {} : Status: Acquiring low-frequency AC light current...'.format(time_count, grad_index, pos_number))
        # After 5 seconds have passed, increase the strobing frequency    
        if jj*t_step == switch_t3:
            light.strobe(shape, frequency_high, amplitude)
            condition = frequency_high
            print('Cycle {} | Grad {} | Loc {} : Status: Acquiring high-frequency AC light current...'.format(time_count, grad_index, pos_number))
        # populate the data time series
        data[ii*n_ts+jj, 0] = np.int(np.floor(ii/n_gradpoints)) #ii  # primary video index
        data[ii*n_ts+jj, 1] = grad_index # gradient index
        data[ii*n_ts+jj, 2] = pos_number # position on gradient index
        data[ii*n_ts+jj, 3] = ts[jj] #this measurement time
        if measurement_type == '4-point':
            data[ii*n_ts+jj, 4] = keithley.source_current # keithley measured current
            data[ii*n_ts+jj, 5] = srs.magnitude*Vpkpk_per_Vrms*V_full_scale/shunt_R #srs output, converted to current
            data[ii*n_ts+jj, 6] = keithley.voltage # keithley measured voltage
        else:
            data[ii*n_ts+jj, 4] = keithley.current # keithley measured current
            data[ii*n_ts+jj, 5] = srs.magnitude*Vpkpk_per_Vrms*V_full_scale/shunt_R #srs output, converted to current
            data[ii*n_ts+jj, 6] = keithley.source_voltage # keithley measured voltage
        data[ii*n_ts+jj, 7] = probe_power # beam intensity
        data[ii*n_ts+jj, 8] = condition # nan if dark, 0 if DC, frequency [Hz] if AC
        data[ii*n_ts+jj, 9] = keithley_too.current # transmittance photodiode Isc
        sleep(t_step)
        
        # For IV sweep:
        #if jj%10 == 0 and jj!=0:
        #    v_idx += 1
        #    keithley.ramp_to_voltage(appl_v[v_idx], steps=2, pause=.02)
        
        # For intensity change:
        #if jj == light_change_idx:
        #    light.set_intensity(probe_power)
    
    # Turn probe off; set light intensity to stress value; remove bias; wait for measurement to finish 
    light.steady()
    light.set_intensity(soak_power)
    keithley.ramp_to_voltage(0, steps=2, pause=.02)
    # Export data
    np.savetxt(csv_save_name, data, delimiter=',')
    elapsed = time()-t
    print('Status: Waiting for next data acquisition cycle...')
    
    
    #UNCOMMENT this line for original version
    #sleep(measurement_time - elapsed)
    
    #####################################################################
    if pos_number == n_gradpoints-1 and not measurement_time2 is None:
        countdown(measurement_time2)
    else:
        countdown(measurement_time - elapsed)
    os.mkdir(os.path.join(save_name, 'temp'))
    ######################################################################
    
     
# END OF THE EXPERIMENT : -----------------------------------------------

#turn equipment off  
keithley.shutdown()
print('Keithley shut.')

#light.off()
print('light NOT turned off')

# plaq.kill()
print("PLAQ process shut")

print('Status: Complete.')
#------------------------------------------------------------------------
