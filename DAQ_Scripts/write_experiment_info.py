# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:46:34 2019

@author: Ryan Stoddard
"""
import json
import os




# =============================================================================
classID = '211214_Swift_Samples'
material_str = ''
expt_str = 'PL_PC_T_DF_FA80Cs20PbI90Br10_1sun_25C_50RH_Air_4'
sampleID = '210517LX8-A2'
# =============================================================================


if material_str == '':
    directory = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\'+ expt_str + '\\Experiment Info\\'
else:
    directory = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\' + material_str + '\\' + expt_str + '\\Experiment Info\\'

data = {'ClassID': classID,
        'ExperimentID': expt_str,
        'SampleID': sampleID,
        'Analysis Date': '12/15/2021',
        'Temperature (deg C)': 25,
        'Atmosphere_RH (%)': 50,
        'Atmosphere_O2 (%)': 21,
        'Atmosphere_N2 (%)': 79,
        'Excitation Intensity': 8, # NSuns as number - probe suns
        'Stress Intensity': 1, # NSuns as number - soak suns
        'Excitation Source': 'Green LED',
        'User': 'Wiley',
        'Ld_contacts': False, # True or False (are they in the frame?)
        'using_XYstage': True, # True or False
        'Ld_data': True,
        'Transmissivity_data': True,
        'channel_length': 2e-4,
        'channel_width': 3e-4,
        'Other Comments': 'none',
        'vidTimeMs': 5000,
        'nrFrames': 50,
        'vidTimeIntervalMs': 5*60*1000,
        'nGradPoints': 1,
        'nGradients': 1,
        'Background_Light': 'none',
        'Encapsulation':'none',
        'Microscope':1,
        #'Objective':'Mitutoyo_100X',
        'Objective':'Olympus_50X_LWD',
        # Options: 'Mitutoyo_50X','Mitutoyo_100X','Olympus_50X','Olympus_50X_LWD','Olympus_5X_LWD'
        'ND06':'out', # neutral density filter status
        'ND25':'in',
        'ND50':'out',
        'Dark_Field': True,
        'Bright_Field': False,
        'Sample Type': 'film', #'film' or 'device'
        'Remarks':'Swift Sample from May 17 Batch',
        } 


if not os.path.exists(directory):
    os.makedirs(directory)

with open(directory + 'experiment_info.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)