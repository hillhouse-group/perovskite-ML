# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:07:06 2019

@author: Ryan Stoddard
"""
import json
import os


#classID = '212308_Gradients_Evan'
#material_str = ''
#expt_str = 'PL_PC_T_FA80Cs20Br17I81SCN020_FA755Cs195PEA050Br17I81SCN020_075MACL_25C_40RH'


# =============================================================================
classID = '211214_Swift_Samples'
material_str = ''
expt_str = 'PL_PC_T_DF_FA80Cs20PbI90Br10_1sun_25C_50RH_Air_4'
sampleID = '210517LX8-A2'
# =============================================================================


if material_str == '':
    directory = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\'+ expt_str + '\\Sample Info\\'
else:
    directory = u'\\\\?\\C:\\Users\\Administrator\\Documents\\Users\\Wiley\\PLVA Measurements\\' + classID + '\\' + material_str + '\\' + expt_str +  '\\Sample Info\\'

data = {"SampleID": sampleID, 
        "Fabrication Date": '05/17/21', 
        "Starting Composition A-site":'FA 0.8 Cs 0.2',
        "Starting Composition B-site": "Pb 1.0",
        "Starting Composition X-site": "I 0.90, Br 0.10",
        "Ending Composition A-site": 'FA 0.8, Cs 0.2', 
        "Ending Composition B-site": "Pb 1.0",
        "Ending Composition X-site": "I 0.90, Br 0.10",
        
        # For Gradients ---------------------------------------------------
        "Fab Method": "Vapor Deposition", 
        "Fab Details": "Swift Proprietary",
        "Ink Solvent": "N/A",
        "Film Thickness, nm": 'Swift Proprietary',
        "Fabrication Environment": "Swift Proprietary", 
        # -----------------------------------------------------------------
        
# =============================================================================
         # For Spin-coated samples,----------------------------------------
        # "Fab Method": "Spin Coating", 
        # "Fab Details": "Check with Yuhuan", 
        # "Ink Solvent": "Check with Yuhuan",
        # "Film Thickness, nm": 300,
        # "Fabrication Environment": "Spin coater GB", 
#         # ----------------------------------------------------------------
# =============================================================================
        
        #"Ink Concentraion, M": "Pb(OAc)2*3H2O 0.59 MAI 1.78",
        "Ink Concentraion, M": "N/A",
        #"Ink Concentraion, M": "1.0 M",
        "Anneal Temp, C": "Swift Proprietary", 
        "Anneal time, min": "Swift Proprietary",
        "Fab O2 ppm": "XX", 
        "Fab H20 ppm": "XX", 
        "Storage Comments": "Big GB", 
        "User": "Swift", 
        #"Other Fabrication Comments": "10 min RT dry before anneal",
        "Other Fabrication Comments":""
        } 

if not os.path.exists(directory):
    os.makedirs(directory)
    
with open(directory + 'sample_info.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
    
    
# Spin Coating Fab details string example: 'XXrpm XXs + XXrpm XXs + toluene XXuL XXs remaining'